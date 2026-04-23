# Task4 Smoke Tests And Migration Notes

## Minimal runnable verification

Run once in the project root:

```powershell
conda run -n finance-agent python scripts/smoke_minimal.py
```

Optional static keyword regression:

```powershell
conda run -n finance-agent python scripts/check_classification_keywords.py
```

This script covers three layers:

- usecase smoke:
  - create material task
  - apply manual correction
  - export excel/text
- retrieval smoke:
  - sqlite upsert/query
  - faiss upsert/query
- graph smoke:
  - material route
  - travel route
  - generic route
- keyword integrity smoke:
  - verify `app/graph/nodes.py` keeps UTF-8 Chinese route/doc-type keywords
  - verify `app/services/validator.py` keeps UTF-8 Chinese expense-category rules

## Modified files and intent

- `scripts/smoke_minimal.py`
  - Add a single minimal smoke runner for usecase/retrieval/graph and include keyword-integrity coverage.
- `scripts/check_classification_keywords.py`
  - Add a static regression check that locks route/category keywords to expected UTF-8 Chinese strings.
- `MIGRATION_TASK4.md`
  - Explain how to run smoke validation and document migration context.

## Recommended call path after refactor

UI layer:

- `streamlit_app.py`
  - only calls usecase modules

Usecase layer:

- `app/usecases/material_agent.py`
- `app/usecases/travel_agent.py`
- `app/usecases/task_orchestration.py`
  - main business orchestration entry

Compatibility layer:

- `app/services/local_runner.py`
  - compatibility wrappers only (delegates to `task_orchestration`)

Retrieval layer:

- `app/retrieval/factory.py`
  - choose backend by `RAG_BACKEND=sqlite|faiss`
- `app/retrieval/sqlite_retriever.py`
  - metadata/content source of truth (SQLite)
- `app/retrieval/faiss_retriever.py`
  - vector index and ANN retrieval only

## Remaining technical debt

- No isolated test DB fixture yet; smoke currently writes real task/case data and then partially cleans temporary retrieval documents.
- Retrieval smoke currently validates hit existence, but not strict ranking consistency between sqlite/faiss.
- Need CI integration (e.g. GitHub Actions) to run `scripts/smoke_minimal.py` automatically.
- Need CI integration (e.g. GitHub Actions) to run `scripts/check_classification_keywords.py` as an encoding/regression guard.
- Usecase API can be further typed (structured return DTOs instead of raw ORM instances).
