# Data analysis towards reusable and composable artificial intelligence for sustainable battery research

Code, frozen inputs and results of the article of the same title.

| Path | Contents |
| --- | --- |
| `reproduce_all.py` | Regenerates all derived tables of `2_analysis/` in dependency order from the frozen inputs and verifies key outputs |
| `1_workflow_framework/` | Composable workflow framework (a Process Coordinator and six module interfaces over a shared candidate record), the Case 1 screening replay and end-to-end chain, and the Case 2 search code with its six configurations |
| `2_analysis/` | Corpus records, text mining, clustering, algorithm detection, data-source coding, scale tests, citation flows, the code and data census with RRL/CRL scoring and cross-scale scan, maps, the Case 2 OEDB snapshot and sharing results, the cost model and the battery-content counts |

## Quick start

1. **Regenerate all derived analysis tables** (Python 3.9 and R 4.5.1):

```bash
python reproduce_all.py
```

This runs all 10 executable analysis sections in dependency order and writes every derived table referenced in `2_analysis/` and in the article's Source Data files (`Source_Data_Fig2–6.xlsx`) into its standard path. You can also run each section individually using the commands in `2_analysis/README.md`.

2. **Workflow framework (Case 1 and Case 2)**: see `1_workflow_framework/README.md` for `reproduce_case1.py` and `battery_workflow/examples/run_case2.py`.

Each folder has its own README with the setup, run commands and the figures and Supplementary Information sections it supports. The 755 analysed articles are listed in Supplementary Data 1 of the article.

## Licence

Code written for this study is under the MIT licence (`LICENSE` in the repository root). Third-party code and data keep their own terms (`2_analysis/THIRD_PARTY_NOTICES.md`); the OEDB CSV snapshot in `2_analysis/10_case2_sharing/oedb_v2026-05-11/` is redistributed unchanged under CC BY 4.0. The three upstream repositories of Case 1 are not redistributed: `1_workflow_framework/tools/fetch_external_data.py` clones them at pinned commits, and values quoted from them in `1_workflow_framework/results/` remain the work of their authors.

## Citation

Please cite the article when using this code (reference to be added on publication).
