from __future__ import annotations

from typing import Any

import streamlit as st


_AGENT_RUNTIME_METRICS_KEY = "agent_runtime_metrics"


def _empty_metric_bucket() -> dict[str, int]:
    return {
        "llm_attempts": 0,
        "llm_hits": 0,
        "action_attempts": 0,
        "action_successes": 0,
    }


def _ensure_metric_bucket(scope: str) -> dict[str, int]:
    metrics = st.session_state.setdefault(_AGENT_RUNTIME_METRICS_KEY, {})
    if not isinstance(metrics, dict):
        metrics = {}
        st.session_state[_AGENT_RUNTIME_METRICS_KEY] = metrics
    key = str(scope or "global").strip().lower() or "global"
    bucket = metrics.get(key)
    if not isinstance(bucket, dict):
        bucket = _empty_metric_bucket()
        metrics[key] = bucket
    for metric_key in _empty_metric_bucket():
        bucket[metric_key] = int(bucket.get(metric_key) or 0)
    return bucket


def record_llm_outcome(scope: str, hit: bool) -> None:
    bucket = _ensure_metric_bucket(scope)
    bucket["llm_attempts"] += 1
    if hit:
        bucket["llm_hits"] += 1


def record_action_outcome(scope: str, ok: bool) -> None:
    bucket = _ensure_metric_bucket(scope)
    bucket["action_attempts"] += 1
    if ok:
        bucket["action_successes"] += 1


def agent_metric_snapshot(scope: str) -> dict[str, Any]:
    bucket = dict(_ensure_metric_bucket(scope))
    llm_attempts = int(bucket.get("llm_attempts") or 0)
    llm_hits = int(bucket.get("llm_hits") or 0)
    action_attempts = int(bucket.get("action_attempts") or 0)
    action_successes = int(bucket.get("action_successes") or 0)
    llm_rate = (llm_hits / llm_attempts * 100) if llm_attempts else 0.0
    action_rate = (action_successes / action_attempts * 100) if action_attempts else 0.0
    return {
        "llm_attempts": llm_attempts,
        "llm_hits": llm_hits,
        "llm_hit_rate": llm_rate,
        "action_attempts": action_attempts,
        "action_successes": action_successes,
        "action_success_rate": action_rate,
    }


def render_agent_metric_caption(scope: str) -> None:
    snapshot = agent_metric_snapshot(scope)
    st.caption(
        "Agent运行指标 · "
        f"LLM命中率 {snapshot['llm_hit_rate']:.1f}% ({snapshot['llm_hits']}/{snapshot['llm_attempts']}) · "
        f"执行成功率 {snapshot['action_success_rate']:.1f}% "
        f"({snapshot['action_successes']}/{snapshot['action_attempts']})"
    )
