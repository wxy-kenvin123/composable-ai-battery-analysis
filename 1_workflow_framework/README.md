# Workflow framework

Workflow framework and Case 1 run files for *Data analysis towards reusable and composable artificial intelligence for sustainable battery research* (Supplementary Notes 7–10, Supplementary Figs. 5 and 6, Supplementary Table 12). Case 2 outputs, the OEDB snapshot and the remaining analysis code are in `../2_analysis/`.

## Structure

| Path | Description |
| --- | --- |
| `battery_workflow/` | Coordinator and six module interfaces over a shared candidate record, plus Case 2 search code (`case_d/`) |
| `battery_workflow/case_d/configs/` | Six self-contained Case 2 configuration files (status: exploratory; frozen before the run) |
| `battery_workflow/case_d/rules/` | Five Literature-rule JSON packs for Case 2 |
| `results/workflow_run_evidence/` | Case 1 screening replay (`438 -> 319 -> 306 -> 56`; trace, funnel, gate decisions, shortlist) |
| `results/htvs_mlsse_continuation/` | ML-SSE executed on the 56 survivors |
| `results/case_a_full/` | End-to-end chain, literature cards and derived tables (Supplementary Figs. 5 and 6) |
| `scripts/` | Value-added, journey and figure scripts; `reproduce_case1.py analysis` writes Supplementary Figs. 5 and 6 |
| `data/census/` | Registry metadata (`registry_tools.csv`, `tool_agent_mapping.csv`) used to label steps in the trace |
| `reproduce_case1.py` | Reruns Case 1 into `rerun_case1/` and compares with `results/` |
| `tools/fetch_external_data.py` | Clones upstream Case 1 repositories at pinned commits and checks SHA-256 |

Code classes and trace keys vs figure module names:

| Code class (trace key) | Figure module |
| --- | --- |
| `KnowledgeMiner` | Knowledge Miner |
| `FormulationDesigner` | Materials Designer |
| `PropertyScreener` (`HTPredictor`, `ExecutedToolScreener`) | HT Predictor |
| `MechanismSimulator` | Mechanism Simulator |
| `ExperimentFeedback` (`SimulationFeedback` when using AIMD) | Experiment Planner |
| `ProcessCoordinator` | Process Coordinator |

## Setup

- Python 3.9 (`requirements.txt`). Unpickling the upstream ML-SSE `model.joblib` requires `scikit-learn==1.0.2` (or set `BATTAGENT_ML_SSE_PYTHON` to an interpreter with `1.0.x`). The AIMD stage uses a separate interpreter with `pymatgen` and `pymatgen-analysis-diffusion`, set via `BATTAGENT_PYMATGEN_PYTHON`.
- No LLM API key is required (`deterministic=True` by default; optional LLM settings can be provided via environment variables documented in `battery_workflow/llm_client.py`).
- Upstream Case 1 repositories have no redistribution licence and are fetched at pinned commits (`core.autocrlf=false`):

```bash
python tools/fetch_external_data.py --cache <DIR>
export BATTAGENT_REPOS_CACHE=<DIR>
```

- Case 2 reads the OEDB snapshot in `../2_analysis/10_case2_sharing/oedb_v2026-05-11/` (CC BY 4.0). Set `BATTAGENT_OEDB_DIR` before running Case 2 scripts.

## Running

Case 1:

```bash
python reproduce_case1.py all   # or: screening | continuation | chain | analysis | compare
```

Case 2 (`battery_workflow/examples/run_case2.py`):

```bash
export BATTAGENT_OEDB_DIR=../2_analysis/10_case2_sharing/oedb_v2026-05-11
python battery_workflow/examples/run_case2.py --config sharing_conditions --seeds 300-301 --workers 4 --outdir <DIR>
```

## Licence

MIT (`../LICENSE`) for code written for this study. Upstream repositories retain their authors' terms. Values quoted from upstream releases in `results/`, such as the screened properties in `results/workflow_run_evidence/table_htvs_gate_decisions.csv`, remain the work of their authors; the tables and run records name the repository and commit each value comes from.
