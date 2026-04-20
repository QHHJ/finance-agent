from __future__ import annotations

import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from app.services import learning

from .dto import OperationResult, TravelStatus


def as_uploaded_list(uploaded_value: Any) -> list[Any]:
    if uploaded_value is None:
        return []
    if isinstance(uploaded_value, list):
        return [item for item in uploaded_value if item is not None]
    return [uploaded_value]


def format_amount(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}"


def sum_profile_amount(profiles: list[dict[str, Any]]) -> float | None:
    numbers = [p.get("amount") for p in profiles if p.get("amount") is not None]
    if not numbers:
        return None
    return float(sum(numbers))


def split_profiles_to_go_return(profiles: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not profiles:
        return [], []
    if len(profiles) == 1:
        return profiles[:], []

    with_date = [p for p in profiles if p.get("date_obj") is not None]
    if with_date:
        unique_dates = sorted({p["date_obj"].date() for p in with_date})
        if len(unique_dates) >= 2:
            split = max(1, len(unique_dates) // 2)
            go_dates = set(unique_dates[:split])
            go: list[dict[str, Any]] = []
            ret: list[dict[str, Any]] = []
            undecided: list[dict[str, Any]] = []
            for profile in profiles:
                date_obj = profile.get("date_obj")
                if date_obj is None:
                    undecided.append(profile)
                    continue
                if date_obj.date() in go_dates:
                    go.append(profile)
                else:
                    ret.append(profile)
            for profile in undecided:
                if len(go) <= len(ret):
                    go.append(profile)
                else:
                    ret.append(profile)
            if not ret and len(go) > 1:
                ret.append(go.pop())
            return go, ret

    ordered = sorted(profiles, key=lambda p: p.get("index", 0))
    split = max(1, len(ordered) // 2)
    if split >= len(ordered):
        split = len(ordered) - 1
    return ordered[:split], ordered[split:]


def split_payment_profiles_to_go_return(
    payments: list[dict[str, Any]],
    go_target: float | None,
    return_target: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not payments:
        return [], []
    if len(payments) == 1:
        return payments[:], []

    can_amount_match = (
        go_target is not None
        and return_target is not None
        and all(p.get("amount") is not None for p in payments)
        and len(payments) <= 18
    )
    if can_amount_match:
        values = [float(p["amount"]) for p in payments]
        total = sum(values)
        n = len(values)
        all_mask = (1 << n) - 1
        best_mask = None
        best_score = float("inf")

        for mask in range(0, all_mask + 1):
            go_sum = 0.0
            for idx in range(n):
                if (mask >> idx) & 1:
                    go_sum += values[idx]
            return_sum = total - go_sum
            score = abs(go_sum - go_target) + abs(return_sum - return_target)
            if mask in (0, all_mask):
                score += 10000.0
            if score < best_score:
                best_score = score
                best_mask = mask

        if best_mask is not None:
            go: list[dict[str, Any]] = []
            ret: list[dict[str, Any]] = []
            for idx, profile in enumerate(payments):
                if (best_mask >> idx) & 1:
                    go.append(profile)
                else:
                    ret.append(profile)
            if go and ret:
                return go, ret

    return split_profiles_to_go_return(payments)


def build_assignment_from_profiles(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    transport_tickets = [p for p in profiles if p.get("doc_type") == "transport_ticket"]
    transport_payments = [p for p in profiles if p.get("doc_type") == "transport_payment"]
    flight_details = [p for p in profiles if p.get("doc_type") == "flight_detail"]
    hotel_invoices = [p for p in profiles if p.get("doc_type") == "hotel_invoice"]
    hotel_payments = [p for p in profiles if p.get("doc_type") == "hotel_payment"]
    hotel_orders = [p for p in profiles if p.get("doc_type") == "hotel_order"]
    unknowns = [p for p in profiles if p.get("doc_type") == "unknown"]

    go_tickets, return_tickets = split_profiles_to_go_return(transport_tickets)
    go_details, return_details = split_profiles_to_go_return(flight_details)

    go_ticket_amount = sum_profile_amount(go_tickets)
    return_ticket_amount = sum_profile_amount(return_tickets)
    go_payments, return_payments = split_payment_profiles_to_go_return(
        transport_payments,
        go_ticket_amount,
        return_ticket_amount,
    )
    go_payment_amount = sum_profile_amount(go_payments)
    return_payment_amount = sum_profile_amount(return_payments)
    hotel_invoice_amount = sum_profile_amount(hotel_invoices)
    hotel_payment_amount = sum_profile_amount(hotel_payments)

    for p in go_tickets:
        p["slot"] = "go_ticket"
    for p in go_payments:
        p["slot"] = "go_payment"
    for p in go_details:
        p["slot"] = "go_detail"
    for p in return_tickets:
        p["slot"] = "return_ticket"
    for p in return_payments:
        p["slot"] = "return_payment"
    for p in return_details:
        p["slot"] = "return_detail"
    for p in hotel_invoices:
        p["slot"] = "hotel_invoice"
    for p in hotel_payments:
        p["slot"] = "hotel_payment"
    for p in hotel_orders:
        p["slot"] = "hotel_order"
    for p in unknowns:
        p["slot"] = "unknown"

    return {
        "go_ticket": [p["file"] for p in go_tickets],
        "go_payment": [p["file"] for p in go_payments],
        "go_detail": [p["file"] for p in go_details],
        "return_ticket": [p["file"] for p in return_tickets],
        "return_payment": [p["file"] for p in return_payments],
        "return_detail": [p["file"] for p in return_details],
        "hotel_invoice": [p["file"] for p in hotel_invoices],
        "hotel_payment": [p["file"] for p in hotel_payments],
        "hotel_order": [p["file"] for p in hotel_orders],
        "unknown": [p["file"] for p in unknowns],
        "go_ticket_amount": go_ticket_amount,
        "go_payment_amount": go_payment_amount,
        "return_ticket_amount": return_ticket_amount,
        "return_payment_amount": return_payment_amount,
        "hotel_invoice_amount": hotel_invoice_amount,
        "hotel_payment_amount": hotel_payment_amount,
    }


def organize_materials(
    pool_files: list[Any],
    *,
    build_profile: Callable[[Any, int], dict[str, Any]],
    manual_overrides: dict[str, str] | None = None,
    apply_overrides: Callable[[list[dict[str, Any]], dict[str, str]], int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    profiles = [build_profile(file, idx) for idx, file in enumerate(pool_files)]
    if manual_overrides and apply_overrides:
        apply_overrides(profiles, manual_overrides)
    assignment = build_assignment_from_profiles(profiles)
    return assignment, profiles


def build_travel_agent_status(assignment: dict[str, Any]) -> dict[str, Any]:
    required_slots = [
        ("go_ticket", "去程机票发票/票据"),
        ("go_payment", "去程支付记录"),
        ("go_detail", "去程机票明细"),
        ("return_ticket", "返程机票发票/票据"),
        ("return_payment", "返程支付记录"),
        ("return_detail", "返程机票明细"),
        ("hotel_invoice", "酒店发票"),
        ("hotel_payment", "酒店支付记录"),
        ("hotel_order", "酒店订单截图"),
    ]
    missing = [label for key, label in required_slots if not as_uploaded_list(assignment.get(key))]

    issues: list[str] = []
    comparisons = [
        ("去程交通", assignment.get("go_ticket_amount"), assignment.get("go_payment_amount")),
        ("返程交通", assignment.get("return_ticket_amount"), assignment.get("return_payment_amount")),
        ("酒店", assignment.get("hotel_invoice_amount"), assignment.get("hotel_payment_amount")),
    ]
    for name, left, right in comparisons:
        if left is None or right is None:
            continue
        if abs(float(left) - float(right)) > 0.01:
            issues.append(f"{name}票据金额与支付记录金额不一致：{format_amount(left)} vs {format_amount(right)}")

    unknown_files = as_uploaded_list(assignment.get("unknown"))
    tips: list[str] = []
    if unknown_files:
        tips.append(f"有 {len(unknown_files)} 份材料尚未识别到明确类型，可在聊天区说明用途后重传。")

    status = TravelStatus(missing=missing, issues=issues, tips=tips, complete=not missing and not issues)
    return status.to_dict()


def merge_uploaded_lists(first: list[Any], second: list[Any]) -> list[Any]:
    merged: list[Any] = []
    seen: set[str] = set()
    for item in list(first) + list(second):
        name = str(getattr(item, "name", ""))
        size = str(getattr(item, "size", ""))
        key = f"{name}:{size}"
        if key in seen:
            continue
        merged.append(item)
        seen.add(key)
    return merged


def sanitize_export_name(name: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", (name or "").strip())
    cleaned = cleaned.strip(" .")
    return cleaned or "差旅报销材料"


def safe_uploaded_filename(name: str, default_stem: str) -> str:
    raw = Path(name or "").name
    if not raw:
        raw = default_stem
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", raw).strip()
    return cleaned or default_stem


def amount_suffix(amount: float | None) -> str:
    if amount is None:
        return "金额未知"
    if abs(amount - round(amount)) <= 0.01:
        return f"{int(round(amount))}元"
    return f"{amount:.2f}元"


def zip_ensure_dir(zip_file: zipfile.ZipFile, dir_path: str) -> None:
    normalized = dir_path.replace("\\", "/").rstrip("/") + "/"
    zip_file.writestr(normalized, b"")


def zip_write_uploaded_files(zip_file: zipfile.ZipFile, target_dir: str, files: list[Any]) -> None:
    zip_ensure_dir(zip_file, target_dir)
    for idx, uploaded in enumerate(files, start=1):
        original_name = str(getattr(uploaded, "name", ""))
        safe_name = safe_uploaded_filename(original_name, f"file_{idx}")
        stored_name = f"{idx:02d}_{safe_name}"
        zip_file.writestr(f"{target_dir}/{stored_name}", uploaded.getvalue())


def build_travel_package_zip(
    package_name: str,
    go_ticket_files: list[Any],
    go_payment_files: list[Any],
    go_detail_files: list[Any],
    return_ticket_files: list[Any],
    return_payment_files: list[Any],
    return_detail_files: list[Any],
    hotel_invoice_files: list[Any],
    hotel_payment_files: list[Any],
    hotel_order_files: list[Any],
    go_ticket_amount: float | None,
    go_payment_amount: float | None,
    return_ticket_amount: float | None,
    return_payment_amount: float | None,
    hotel_invoice_amount: float | None,
    hotel_payment_amount: float | None,
) -> bytes:
    root_name = sanitize_export_name(package_name)
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        go_root = f"{root_name}/出差去程交通报销"
        zip_write_uploaded_files(zip_file, f"{go_root}/去程机票发票_{amount_suffix(go_ticket_amount)}", go_ticket_files)
        zip_write_uploaded_files(zip_file, f"{go_root}/去程支付记录_{amount_suffix(go_payment_amount)}", go_payment_files)
        zip_write_uploaded_files(zip_file, f"{go_root}/去程机票明细", go_detail_files)

        return_root = f"{root_name}/出差返程交通报销"
        zip_write_uploaded_files(
            zip_file,
            f"{return_root}/返程机票发票_{amount_suffix(return_ticket_amount)}",
            return_ticket_files,
        )
        zip_write_uploaded_files(
            zip_file,
            f"{return_root}/返程支付记录_{amount_suffix(return_payment_amount)}",
            return_payment_files,
        )
        zip_write_uploaded_files(zip_file, f"{return_root}/返程机票明细", return_detail_files)

        hotel_root = f"{root_name}/酒店报销"
        zip_write_uploaded_files(zip_file, f"{hotel_root}/酒店发票_{amount_suffix(hotel_invoice_amount)}", hotel_invoice_files)
        zip_write_uploaded_files(zip_file, f"{hotel_root}/支付记录_{amount_suffix(hotel_payment_amount)}", hotel_payment_files)
        zip_write_uploaded_files(zip_file, f"{hotel_root}/订单截图", hotel_order_files)
    buffer.seek(0)
    return buffer.getvalue()


def learn_from_profiles(profiles: list[dict[str, Any]], assignment: dict[str, Any], reason: str) -> OperationResult:
    try:
        learning.learn_from_travel_profiles(profiles, assignment, reason=reason)
        return OperationResult(ok=True)
    except Exception as exc:
        return OperationResult(ok=False, message=str(exc))
