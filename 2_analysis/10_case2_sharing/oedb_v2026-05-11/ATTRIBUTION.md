# Open Electrolyte Database for Batteries (OEDB), v2026-05-11

The two CSV files in this folder are the OEDB computational-database snapshot used in Case 2, redistributed unchanged
under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

Source: https://oedb.jp/ (MCP endpoint https://oedb.jp/mcp). Reference: Nakamura, K. et al. Open electrolyte database
generated via an automated molecular dynamics simulation framework. npj Comput. Mater. 12, 215 (2026),
https://doi.org/10.1038/s41524-026-02093-y.

Contents (5,616 rows, unfiltered and unmodified): `oedb_mcp_exportData_v2026-05-11.csv` (extracted CSV payload from
MCP server 2.0.0 `exportData`, primary input read by the Case 2 workflow) and `oedb-electrolytes-v2026-05-11.csv`
(website CSV download referenced in Source Data Fig. 6c). Redundant raw JSON/Arrow serializations of the same 5,616
rows are omitted to keep the repository lightweight.

The workflow framework (`1_workflow_framework/`) verifies the per-file SHA-256 checksums of both CSVs before loading;
set `BATTAGENT_OEDB_DIR` to this folder.
