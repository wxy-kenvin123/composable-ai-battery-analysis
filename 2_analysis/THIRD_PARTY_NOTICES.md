# Third-party code and data

| Component | Location | Source | Licence |
|---|---|---|---|
| Noun-extraction pipeline (upstream reference) | Referenced in `02_text_mining/` (`wordbags_to_counts.R` consumes its outputs) | Gohr, C. et al. *Nat. Sustain.* 8, 970–978 (2025), https://doi.org/10.1038/s41893-025-01598-6 | MIT (upstream script not duplicated) |
| Citation-network matrix and diagonal-benchmark helpers | Inlined with attribution in `07_citation_flows/scripts/compute_flows.py` (`build_raw_matrix`, `diagonal_benchmark`, `benchmark_matrix`) | Hemmelder, A. et al. *Nat. Energy* 11, 313–323 (2026), https://doi.org/10.1038/s41560-026-01985-z | MIT |
| Open Electrolyte Database for Batteries (OEDB, v2026-05-11) | `10_case2_sharing/oedb_v2026-05-11/` (MCP export CSV, website CSV and `ATTRIBUTION.md`) | Nakamura, K. et al. *npj Comput. Mater.* 12, 215 (2026), https://doi.org/10.1038/s41524-026-02093-y; https://oedb.jp/ | CC BY 4.0, unchanged (`ATTRIBUTION.md`) |
| OpenAlex topic/category summaries | `07_citation_flows/outside_corpus/output/` | OpenAlex API (retrieved November 2026) | CC0 |
| Bibliographic identifiers (title, journal, year, DOI) | `04_algorithms/data/articles755.csv`, `07_citation_flows/data/corpus.csv` | Scopus exports and publisher records | Bibliographic facts only (no abstracts, keywords, affiliations or reference strings) |

Map outlines in `09_maps/` are loaded at run time from Natural Earth (public domain) via `rnaturalearth`.
