from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _u(value: str) -> str:
    return value.encode("ascii").decode("unicode_escape")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _load_module(relative_path: str) -> tuple[str, ast.Module]:
    source = (ROOT / relative_path).read_text(encoding="utf-8", errors="strict")
    return source, ast.parse(source)


def _find_assignment(module: ast.Module, name: str) -> object:
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == name and node.value is not None:
                return ast.literal_eval(node.value)
    raise AssertionError(f"assignment not found: {name}")


def _find_function(module: ast.Module, name: str) -> ast.FunctionDef:
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function not found: {name}")


def _collect_string_literals(node: ast.AST) -> set[str]:
    values: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            values.add(child.value)
    return values


def _assert_contains_all(actual: set[str], expected: list[str], label: str) -> None:
    missing = [item for item in expected if item not in actual]
    _assert(not missing, f"{label} missing expected strings: {missing}")


def run_keyword_integrity_checks() -> dict[str, int]:
    nodes_source, nodes_module = _load_module("app/graph/nodes.py")
    validator_source, validator_module = _load_module("app/services/validator.py")

    _assert(_u("\\u5dee\\u65c5") in nodes_source, "nodes.py missing travel keyword text")
    _assert(_u("\\u5236\\u5ea6") in nodes_source, "nodes.py missing policy keyword text")
    _assert(_u("\\u6750\\u6599") in nodes_source, "nodes.py missing material keyword text")

    travel_keywords = set(_find_assignment(nodes_module, "TRAVEL_KEYWORDS"))
    _assert_contains_all(
        travel_keywords,
        [
            _u("\\u5dee\\u65c5"),
            _u("\\u51fa\\u5dee"),
            _u("\\u673a\\u7968"),
            _u("\\u9ad8\\u94c1"),
            _u("\\u706b\\u8f66"),
            _u("\\u9152\\u5e97"),
            _u("\\u884c\\u7a0b"),
        ],
        "TRAVEL_KEYWORDS",
    )

    policy_keywords = set(_find_assignment(nodes_module, "POLICY_KEYWORDS"))
    _assert_contains_all(
        policy_keywords,
        [
            _u("\\u5236\\u5ea6"),
            _u("\\u89c4\\u5219"),
            _u("\\u89c4\\u5b9a"),
            _u("\\u653f\\u7b56"),
            _u("\\u62a5\\u9500\\u6807\\u51c6"),
            _u("\\u53ef\\u4ee5\\u62a5\\u9500"),
        ],
        "POLICY_KEYWORDS",
    )

    material_keywords = set(_find_assignment(nodes_module, "MATERIAL_KEYWORDS"))
    _assert_contains_all(
        material_keywords,
        [
            _u("\\u6750\\u6599"),
            _u("\\u5165\\u5e93"),
            _u("\\u89c4\\u683c"),
            _u("\\u578b\\u53f7"),
            _u("\\u6570\\u91cf"),
            _u("\\u5355\\u4f4d"),
        ],
        "MATERIAL_KEYWORDS",
    )

    travel_doc_literals = _collect_string_literals(_find_function(nodes_module, "_guess_travel_doc_type"))
    _assert_contains_all(
        travel_doc_literals,
        [
            _u("\\u9152\\u5e97"),
            _u("\\u4f4f\\u5bbf"),
            _u("\\u643a\\u7a0b"),
            _u("\\u98de\\u732a"),
            _u("\\u673a\\u7968"),
            _u("\\u9ad8\\u94c1"),
            _u("\\u6253\\u8f66"),
            _u("\\u51fa\\u79df\\u8f66"),
        ],
        "_guess_travel_doc_type",
    )

    travel_assign_literals = _collect_string_literals(_find_function(nodes_module, "travel_assign_node"))
    _assert_contains_all(
        travel_assign_literals,
        [
            _u("\\u6253\\u8f66"),
            _u("\\u6ef4\\u6ef4"),
            _u("\\u7f51\\u7ea6\\u8f66"),
            _u("\\u9ad8\\u94c1"),
            _u("\\u706b\\u8f66"),
            _u("\\u94c1\\u8def"),
            _u("\\u673a\\u7968"),
            _u("\\u822a\\u73ed"),
            _u("\\u822a\\u7a7a"),
        ],
        "travel_assign_node",
    )

    category_rules = _find_assignment(validator_module, "CATEGORY_RULES")
    categories = {str(item["category"]) for item in category_rules if isinstance(item, dict)}
    _assert_contains_all(
        categories,
        [
            _u("\\u5dee\\u65c5\\u8d39"),
            _u("\\u6750\\u6599\\u8d39"),
            _u("\\u529e\\u516c\\u8d39"),
            _u("\\u4e1a\\u52a1\\u62db\\u5f85\\u8d39"),
            _u("\\u57f9\\u8bad\\u8d39"),
            _u("\\u8f6f\\u4ef6\\u670d\\u52a1\\u8d39"),
        ],
        "CATEGORY_RULES",
    )

    suggest_literals = _collect_string_literals(_find_function(validator_module, "suggest_processing"))
    _assert_contains_all(
        suggest_literals,
        [
            _u("\\u5dee\\u65c5\\u8d39"),
            _u("\\u6750\\u6599\\u8d39"),
            _u("\\u884c\\u7a0b\\u5355"),
            _u("\\u9152\\u5e97\\u8ba2\\u5355"),
            _u("\\u5165\\u5e93"),
            _u("\\u89c4\\u683c\\u578b\\u53f7"),
        ],
        "suggest_processing",
    )

    return {
        "node_keyword_groups": 5,
        "travel_doc_checks": 8,
        "travel_assign_checks": 9,
        "validator_category_checks": 6,
        "validator_reason_checks": 6,
    }


def main() -> int:
    summary = run_keyword_integrity_checks()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("[OK] classification keyword integrity checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
