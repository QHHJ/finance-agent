from __future__ import annotations

import argparse
import base64
import json
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.runtime import init_runtime


def _default_base_url() -> str:
    import os

    return str(os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")


def _default_vl_model() -> str:
    import os

    return str(os.getenv("OLLAMA_VL_MODEL") or os.getenv("OLLAMA_MODEL") or "qwen2.5vl:3b")


def _parse_json_loose(text: str) -> dict[str, Any] | None:
    source = str(text or "").strip()
    if not source:
        return None
    try:
        parsed = json.loads(source)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", source)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _encode_images_from_path(file_path: Path, pdf_max_pages: int = 2) -> tuple[list[str], str]:
    suffix = file_path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        data = file_path.read_bytes()
        return [base64.b64encode(data).decode("utf-8")], "image"

    if suffix == ".pdf":
        encoded_pages: list[str] = []
        with fitz.open(file_path) as doc:
            page_count = min(len(doc), max(1, int(pdf_max_pages)))
            for page_idx in range(page_count):
                page = doc.load_page(page_idx)
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                png_bytes = pix.tobytes("png")
                encoded_pages.append(base64.b64encode(png_bytes).decode("utf-8"))
        return encoded_pages, "pdf"

    raise ValueError(f"不支持的文件类型: {file_path.suffix}")


def _ollama_chat_with_images(
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    images: list[str],
    history: list[dict[str, str]] | None = None,
    timeout_sec: int = 90,
) -> str:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for item in list(history or [])[-8:]:
        role = str(item.get("role") or "")
        content = str(item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content[:1500]})

    user_message: dict[str, Any] = {"role": "user", "content": user_prompt}
    if images:
        user_message["images"] = images
    messages.append(user_message)

    try:
        payload = {
            "model": model,
            "stream": False,
            "messages": messages,
            "options": {"temperature": 0.2},
        }
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=(8, timeout_sec))
        resp.raise_for_status()
        content = str((resp.json().get("message") or {}).get("content") or "").strip()
        if content:
            return content
    except Exception:
        pass

    prompt_lines: list[str] = [f"[system] {system_prompt}"]
    for item in messages[1:]:
        prompt_lines.append(f"[{item.get('role')}] {item.get('content')}")
    fallback_prompt = "\n\n".join(prompt_lines)

    payload = {
        "model": model,
        "stream": False,
        "prompt": fallback_prompt,
        "images": images if images else None,
        "options": {"temperature": 0.2},
    }
    resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=(8, timeout_sec))
    resp.raise_for_status()
    return str(resp.json().get("response") or "").strip()


def _build_extraction_prompt() -> str:
    return (
        "请提取这张（或这些）发票里的关键信息，只返回JSON对象，不要额外解释。\n"
        "字段包括：invoice_type, invoice_code, invoice_number, invoice_date, seller, buyer, total_amount, tax_amount, "
        "amount_without_tax, items(数组，元素字段：name,spec,quantity,unit,amount), confidence, notes。"
    )


@dataclass
class FileContext:
    path: Path
    images: list[str]
    file_kind: str


def _open_file_context(file_path: str, pdf_max_pages: int) -> FileContext:
    path = Path(file_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")
    images, file_kind = _encode_images_from_path(path, pdf_max_pages=pdf_max_pages)
    return FileContext(path=path, images=images, file_kind=file_kind)


def _extract_invoice_with_vl(
    *,
    context: FileContext,
    base_url: str,
    model: str,
) -> tuple[dict[str, Any] | None, str]:
    system_prompt = "你是发票信息抽取助手，必须忠于图片内容，不确定就写到notes。"
    raw = _ollama_chat_with_images(
        base_url=base_url,
        model=model,
        system_prompt=system_prompt,
        user_prompt=_build_extraction_prompt(),
        images=context.images,
        history=None,
    )
    parsed = _parse_json_loose(raw)
    return parsed, raw


def _answer_question_with_vl(
    *,
    context: FileContext,
    question: str,
    base_url: str,
    model: str,
    history: list[dict[str, str]],
) -> str:
    system_prompt = (
        "你是发票问答助手。回答要简洁，严格基于发票内容；"
        "不要编造，如果看不清请明确说看不清并指出需要补什么。"
    )
    user_prompt = f"基于当前发票内容回答：{question}"
    return _ollama_chat_with_images(
        base_url=base_url,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=context.images,
        history=history,
    )


def _print_help() -> None:
    print("可用命令：")
    print("  看文件 <路径>      打开/切换文件并自动提取发票字段")
    print("  /file <路径>       同上")
    print("  提取               对当前文件重新提取")
    print("  /extract           同上")
    print("  退出 | quit | exit 结束")
    print("  其它任意输入       对当前文件进行问答")


def _extract_file_path_from_command(user_input: str) -> str:
    text = str(user_input or "").strip()
    if text.startswith("看文件 "):
        return text[len("看文件 ") :].strip()
    if text.startswith("/file "):
        return text[len("/file ") :].strip()
    return ""


def test_qwen25vl_invoice(
    initial_file: str | None = None,
    *,
    model: str | None = None,
    base_url: str | None = None,
    pdf_max_pages: int = 2,
    one_shot_question: str | None = None,
) -> None:
    """
    终端测试函数：调用 qwen2.5vl:3b 抽取发票并支持连续问答。
    """
    init_runtime()
    resolved_model = str(model or _default_vl_model())
    resolved_base_url = str(base_url or _default_base_url()).rstrip("/")

    print(f"[Invoice VL Test] base_url={resolved_base_url}")
    print(f"[Invoice VL Test] model={resolved_model}")

    context: FileContext | None = None
    history: list[dict[str, str]] = []

    def load_file_and_extract(path_text: str) -> None:
        nonlocal context
        context = _open_file_context(path_text, pdf_max_pages=pdf_max_pages)
        print(f"\n已打开文件: {context.path}")
        print(f"文件类型: {context.file_kind}，用于识别的图像页数: {len(context.images)}")
        parsed, raw = _extract_invoice_with_vl(context=context, base_url=resolved_base_url, model=resolved_model)
        print("\n提取结果:")
        if parsed is not None:
            print(json.dumps(parsed, ensure_ascii=False, indent=2))
        else:
            print(raw)

    if initial_file:
        load_file_and_extract(initial_file)

    if one_shot_question:
        if context is None:
            raise ValueError("one_shot_question 模式下必须提供 --file。")
        answer = _answer_question_with_vl(
            context=context,
            question=str(one_shot_question),
            base_url=resolved_base_url,
            model=resolved_model,
            history=history,
        )
        print("\n回答:")
        print(answer)
        return

    print("\n进入终端对话。")
    _print_help()
    while True:
        try:
            user_input = input("\n你> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n已退出。")
            break

        if not user_input:
            continue
        lowered = user_input.lower()
        if lowered in {"退出", "quit", "exit"}:
            print("已退出。")
            break
        if lowered in {"/help", "help", "帮助"}:
            _print_help()
            continue

        path_text = _extract_file_path_from_command(user_input)
        if path_text:
            try:
                # Allow quoted path with spaces.
                maybe_tokens = shlex.split(path_text)
                normalized_path = maybe_tokens[0] if maybe_tokens else path_text
                load_file_and_extract(normalized_path)
            except Exception as exc:
                print(f"打开文件失败: {exc}")
            continue

        if lowered in {"提取", "/extract"}:
            if context is None:
                print("请先用“看文件 <路径>”打开发票。")
                continue
            try:
                parsed, raw = _extract_invoice_with_vl(context=context, base_url=resolved_base_url, model=resolved_model)
                print("\n提取结果:")
                if parsed is not None:
                    print(json.dumps(parsed, ensure_ascii=False, indent=2))
                else:
                    print(raw)
            except Exception as exc:
                print(f"提取失败: {exc}")
            continue

        if context is None:
            print("请先用“看文件 <路径>”打开发票，再提问。")
            continue

        try:
            answer = _answer_question_with_vl(
                context=context,
                question=user_input,
                base_url=resolved_base_url,
                model=resolved_model,
                history=history,
            )
            print(f"\n助手> {answer}")
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": answer})
        except Exception as exc:
            print(f"问答失败: {exc}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="qwen2.5vl:3b 发票提取终端测试")
    parser.add_argument("--file", dest="file_path", default="", help="发票文件路径（图片或PDF）")
    parser.add_argument("--ask", dest="ask", default="", help="单次提问；与 --file 一起使用，执行后直接退出")
    parser.add_argument("--model", dest="model", default="", help="覆盖模型名，默认取 OLLAMA_VL_MODEL 或 qwen2.5vl:3b")
    parser.add_argument("--base-url", dest="base_url", default="", help="覆盖 Ollama 地址，默认取 OLLAMA_BASE_URL")
    parser.add_argument("--pdf-max-pages", dest="pdf_max_pages", type=int, default=2, help="PDF最多渲染页数")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    test_qwen25vl_invoice(
        initial_file=args.file_path or None,
        model=args.model or None,
        base_url=args.base_url or None,
        pdf_max_pages=max(1, int(args.pdf_max_pages or 2)),
        one_shot_question=args.ask or None,
    )
