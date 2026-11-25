# Analysis

Analysis scripts and frozen inputs for *Data analysis towards reusable and composable artificial intelligence for sustainable battery research*. The workflow framework and Case 1 files are in `../1_workflow_framework/`; the 755 corpus articles are listed in Supplementary Data 1.

## Layout

All frozen non-regenerable inputs, the OEDB CSV snapshot and the 60-seed Case 2 result checkpoints are stored directly in their section folders. Run `python ../reproduce_all.py` from this folder (or `python reproduce_all.py` from the repository root) to regenerate all derived tables in place.

| Folder | Contents |
| --- | --- |
| `01_corpus/` | Search strings and annual Scopus records (Methods; Fig. 2a; Fig. S1) |
| `02_text_mining/` | Document-term counter (Methods; upstream PDF noun-extraction script from Gohr et al., *Nat. Sustain.* 8, 970–978, 2025, DOI `10.1038/s41893-025-01598-6`) |
| `03_clustering/` | Ward clustering, IndVal, DCA and choice of K (Fig. 2b–d; Note S1, Table S2, Figs. S2, S3) |
| `04_algorithms/` | Algorithm, AI-type and task detections and summaries (Figs. 2g,h, 3, 4a; Note S1) |
| `05_data_sources/` | Data-source coding and summaries (Fig. 2e; Table S3) |
| `06_scale_tests/` | DCA1 positions, split tests and disconnect test (Fig. 2c; Note S1) |
| `07_citation_flows/` | Citation flows and outside-corpus summaries (Fig. 4b–d; Note S2, Table S4) |
| `08_census/` | Code/data census, RRL/CRL scoring code, cross-scale scan and witnesses (Fig. 5; Notes S3–S6, Tables S5–S12) |
| `09_maps/` | Author locations and external database hosts (Fig. 2e,f) |
| `10_case2_sharing/` | Case 2 OEDB snapshot, sharing runs and main ablation trajectory (Fig. 6c–g; Note S10, Tables S14–S16) |
| `11_tea/` | Record-sharing cost and payback model (Fig. 6h,i; Note S11, Tables S17–S20, Fig. S7) |
| `12_battery_content/` | Battery component, electrolyte and ion term counter and summary (Methods; Note S1, Fig. S4) |

## Running

To regenerate every derived table in dependency order from the repository root:

```bash
python reproduce_all.py
```

Alternatively, run each step from `2_analysis/` in order (`04_algorithms` before `03_clustering/k_choice`, `03_clustering` before `06_scale_tests`, `05_data_sources` before `06_scale_tests` and `09_maps`). Full-text extraction (`02_text_mining/wordbags_to_counts.R`, `04_algorithms/source_original/compute_top3_algorithms.py`, `trim_references.py`), `07_citation_flows/outside_corpus/scripts/`, `08_census/scoring_code/` and `12_battery_content/scripts/count_mentions.py` require non-redistributed full texts, raw Scopus/Crossref exports or upstream repository clones and are included as reference code beside their frozen outputs.

```bash
(cd 01_corpus && Rscript annual_records/nightingale_rose_2026_split.R --svg-only)
(cd 04_algorithms && python -B scripts/aggregate_from_hits.py --outdir data)
(cd 03_clustering && Rscript code/cluster.R output && Rscript code/adapt_model.R output && python -I k_choice/verify_k8_ruler.py)
(cd 05_data_sources && python -B scripts/recount_data_sources.py --write-data)
(cd 06_scale_tests && Rscript scripts/position.R output && python -B scripts/split_tests.py output/data output/data_final_coding && Rscript scripts/dca_axes.R output && Rscript scripts/control_test.R output/control_test)
(cd 07_citation_flows && python -B scripts/compute_flows.py --outdir data)
(cd 08_census && python -B scripts/recount_census.py --outdir data)
(cd 09_maps && Rscript scripts/plot_datahub_map.R output)
(cd 10_case2_sharing && python -I analysis/ablation_readouts.py)
(cd 11_tea && python -B scripts/recompute_tea_ladder.py output)
```

To rerun Case 2 sharing seeds with `../1_workflow_framework/` from this folder (`BATTAGENT_OEDB_DIR` set to `10_case2_sharing/oedb_v2026-05-11`):

```bash
python ../1_workflow_framework/battery_workflow/examples/run_case2.py --config sharing_conditions --seeds 300-301 --workers 4 --outdir <DIR>
python 10_case2_sharing/analysis/compare_sharing_rerun.py <DIR>
```

## Environment & Licence

- **Python 3.9**: `numpy`, `pandas`, `scipy`, `matplotlib`, `openpyxl`, `scikit-learn`, `pyarrow`.
- **R 4.5.1**: `cluster`, `vegan`, `labdsv`, `ggplot2`, `patchwork`, `svglite`, `dplyr`, `tidyr`, `readr`, `sf`, `rnaturalearth`, `ggnewscale`.
- Author code is MIT (`../LICENSE`); third-party components keep their terms (`THIRD_PARTY_NOTICES.md`).
