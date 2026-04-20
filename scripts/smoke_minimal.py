from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4


def _bootstrap_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


@dataclass(slots=True)
class SmokeSummary:
    usecase_task_id: str
    usecase_excel: str
    usecase_text: str
    sqlite_hits: int
    faiss_hits: int
    graph_routes: dict[str, str]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _build_minimal_pdf_bytes(text: str) -> bytes:
    try:
        import fitz  # pymupdf
    except Exception as exc:
        raise RuntimeError("smoke test requires pymupdf/fitz") from exc

    doc = fitz.open()
    try:
        page = doc.new_page()
        page.insert_text((72, 72), text)
        return doc.tobytes()
    finally:
        doc.close()


def run_usecase_smoke() -> tuple[str, str, str]:
    from app.usecases import task_orchestration as task_ops

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"smoke_material_{now}.pdf"
    pdf_bytes = _build_minimal_pdf_bytes("材料费 smoke test 机票 酒店 入库")

    task = task_ops.create_and_process_task(filename, pdf_bytes, auto_process=True, auto_export=True)
    _assert(task is not None, "create_and_process_task returned None")

    latest = task_ops.get_task(task.id)
    _assert(latest is not None, "get_task returned None")

    extracted = dict(latest.extracted_data or {})
    if not isinstance(extracted.get("line_items"), list):
        extracted["line_items"] = []
    if not extracted["line_items"]:
        extracted["line_items"] = [
            {
                "item_name": "烟雾测试项目",
                "spec": "SMOKE-1",
                "quantity": "1",
                "unit": "件",
                "line_total_with_tax": "100.00",
            }
        ]
    extracted["amount"] = str(extracted.get("amount") or "100.00")
    extracted["bill_type"] = str(extracted.get("bill_type") or "增值税普通发票")

    corrected = task_ops.apply_corrections(
        latest.id,
        {
            "expense_category": "材料费",
            "extracted_fields": extracted,
        },
    )
    _assert(corrected is not None, "apply_corrections returned None")

    export_paths = task_ops.export_task(latest.id, export_format="both")
    excel_path = str(export_paths.get("excel_path") or "")
    text_path = str(export_paths.get("text_path") or "")
    _assert(bool(excel_path), "excel export path is empty")
    _assert(bool(text_path), "text export path is empty")
    _assert(Path(excel_path).exists(), f"excel export missing: {excel_path}")
    _assert(Path(text_path).exists(), f"text export missing: {text_path}")
    return latest.id, excel_path, text_path


def run_retrieval_smoke() -> tuple[int, int]:
    from app.retrieval.factory import clear_retriever_cache, get_retriever, rebuild_faiss_index

    payload_text = "差旅 报销 机票 酒店 材料 规则 smoke"

    def _one_backend(backend: str) -> int:
        clear_retriever_cache()
        retriever = get_retriever(backend)
        source_id = f"smoke_{backend}_{uuid4().hex[:8]}"
        doc_key = f"travel_case:{source_id}:1"
        inserted = retriever.upsert_documents(
            source_type="travel_case",
            source_id=source_id,
            documents=[
                {
                    "doc_key": doc_key,
                    "title": "smoke",
                    "content": payload_text,
                    "metadata": {"smoke": True, "backend": backend},
                }
            ],
        )
        _assert(inserted >= 1, f"{backend} upsert failed")
        hits = retriever.query_documents(
            query="机票 报销 酒店",
            source_types=["travel_case"],
            top_k=5,
            min_score=-1.0,
            metadata_filter={"smoke": True, "backend": backend},
        )
        _assert(len(hits) >= 1, f"{backend} query returned no hits")
        retriever.delete_documents(source_type="travel_case", source_id=source_id)
        return len(hits)

    sqlite_hits = _one_backend("sqlite")
    # Ensure FAISS index is present before faiss query.
    rebuild_faiss_index()
    faiss_hits = _one_backend("faiss")
    return sqlite_hits, faiss_hits


def run_graph_smoke() -> dict[str, str]:
    from app.graph.build_graph import build_finance_graph, route_task_type

    app = build_finance_graph()
    _assert(app is not None, "build_finance_graph returned None")

    routes = {
        "material": route_task_type({"task_type": "material"}),
        "travel": route_task_type({"task_type": "travel"}),
        "generic": route_task_type({"task_type": "generic"}),
    }
    _assert(routes["material"] == "material_prepare", "material route mismatch")
    _assert(routes["travel"] == "travel_prepare", "travel route mismatch")
    _assert(routes["generic"] == "generic_suggest", "generic route mismatch")
    return routes


def main() -> int:
    _bootstrap_path()
    task_id, excel_path, text_path = run_usecase_smoke()
    sqlite_hits, faiss_hits = run_retrieval_smoke()
    graph_routes = run_graph_smoke()

    summary = SmokeSummary(
        usecase_task_id=task_id,
        usecase_excel=excel_path,
        usecase_text=text_path,
        sqlite_hits=sqlite_hits,
        faiss_hits=faiss_hits,
        graph_routes=graph_routes,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    print("[OK] smoke_minimal passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
