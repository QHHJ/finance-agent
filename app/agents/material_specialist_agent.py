from __future__ import annotations

from app.usecases import material_agent as material_usecase

from .base import BaseAgent
from .contracts import AgentTask


class MaterialSpecialistAgent(BaseAgent):
    name = "material_specialist_agent"

    def run(self, task: AgentTask):
        objective = str(task.objective or "").strip()
        payload = dict(task.payload or {})

        if objective == "extract_fields":
            extracted = material_usecase.extract_fields(payload.get("task"))
            return self._result(
                summary="Material fields extracted.",
                payload={"fields": extracted},
                events=[self._event("material_fields_extracted", "Material fields extracted.")],
            )

        if objective == "run_llm_fix":
            ok, message, updated_task, updated_fields = material_usecase.run_llm_fix(
                payload.get("task"),
                dict(payload.get("fields") or {}),
            )
            return self._result(
                ok=bool(ok),
                summary=message or ("Material LLM fix completed." if ok else "Material LLM fix failed."),
                payload={"task": updated_task, "fields": updated_fields},
                events=[self._event("material_llm_fix_ran", message or "Material LLM fix completed.")],
            )

        if objective == "auto_split_rows":
            rows, changed = material_usecase.auto_split_rows(payload.get("rows") or [])
            return self._result(
                summary=f"Material rows auto-split: {changed}",
                payload={"rows": rows, "changed": changed},
            )

        if objective == "build_review_compare_rows":
            left_rows, right_rows = material_usecase.build_review_compare_rows(dict(payload.get("fields") or {}))
            return self._result(
                summary=f"Review compare rows built: {len(right_rows)}",
                payload={"left_rows": left_rows, "right_rows": right_rows},
            )

        if objective == "build_rule_llm_compare_rows":
            left_rows, right_rows = material_usecase.build_rule_llm_compare_rows(dict(payload.get("fields") or {}))
            return self._result(
                summary=f"Rule/LLM compare rows built: {len(right_rows)}",
                payload={"left_rows": left_rows, "right_rows": right_rows},
            )

        if objective == "rule_llm_diff_count":
            diff_count = material_usecase.rule_llm_diff_count(dict(payload.get("fields") or {}))
            return self._result(
                summary=f"Rule/LLM diff count: {diff_count}",
                payload={"diff_count": diff_count},
            )

        if objective == "extract_invoice_fields":
            fields = material_usecase.extract_invoice_fields(
                raw_text=str(payload.get("raw_text") or ""),
                pdf_path=payload.get("pdf_path"),
            )
            return self._result(
                summary="Invoice fields extracted.",
                payload={"invoice_fields": fields},
            )

        if objective == "build_material_references":
            references = material_usecase.build_material_references(
                dict(payload.get("fields") or {}),
                str(payload.get("raw_text") or ""),
            )
            return self._result(
                summary="Material references built.",
                payload={"references": references},
            )

        return self._result(ok=False, summary=f"Unsupported material objective: {objective}")
