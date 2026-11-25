# Search strings

## Baseline search (Scopus, October 2025)

Three Scopus queries, one for each battery component term (`electrolyte`, `anode`, `cathode`), combined AI method
terms (`"machine learning"`, `"deep learning"`, `"active learning"`, `"large language model*"`, `"neural network*"`)
with the component term and `battery` or `batteries`, restricted to English-language journal articles and reviews
(1,984 records retrieved; 723 retained after de-duplication, abstract screening and the vocabulary-size audit).

## 2026 search (Scopus, 19 September 2026)

```
TITLE-ABS-KEY(("machine learning" OR "deep learning" OR "active learning" OR "large language model*"
  OR "neural network*" OR "artificial intelligence") AND electrolyte)
AND ALL(battery OR batteries)
AND (DOCTYPE(ar) OR DOCTYPE(re)) AND LANGUAGE(english) AND SRCTYPE(j)
AND PUBYEAR AFT 2025 AND PUBYEAR BEF 2027
```

run with `electrolyte`, `anode` and `cathode` in turn. Applied to completed years up to 2024, this query recalls
94.9% (electrolyte), 91.7% (anode) and 90.7% (cathode) of the baseline records. For 2026 it returned 561
(electrolyte), 230 (anode) and 277 (cathode) records, 1,068 rows in `scopus_2026_hits.csv` (stream, Scopus EID, DOI,
year; Scopus abstracts, keywords and affiliations are not redistributed).

Without the year limit, the same query gives the 2026 counts of Fig. 2a; the counts up to 2025 come from the
October 2025 exports (`../annual_records/`).
