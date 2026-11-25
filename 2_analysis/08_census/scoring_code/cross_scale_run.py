"""
Bounded cross-scale scan over this run’s deduplicated primary resources.

This adaptation counts identifier candidates under the original matching rules.
It does not establish that tools can be chained or count confirmed scientific
links. Source functions and matching constants are preserved from the reference.

For every CSV/TSV header in every audited repository it asks which scale a
property reports -- material, interface, or cell -- and, within a bounded prefix
of at most 20,000 rows or 4 MiB per file, which candidate each row describes. It
then reports, per tool:

  * how many of the three scales the tool's released data covers at all, and
  * whether any single candidate in that tool's data carries properties from
    more than one scale.

The second measure is the one that matters. A tool can publish a material-level
table and a cell-level table and still be unchainable if the two never describe
the same candidate, which is exactly the join a downstream workflow needs. It is
reported twice: once for any cell-level column, and once restricted to cells
that were actually cycled, because a theoretical capacity computed from the same
DFT record as the voltage sits on the far side of the scale boundary without
carrying any new information across it.

A secondary table reports pairwise identifier overlap between repositories.
Identifiers are compared as strings (with chemical formulas normalised to
reduced element counts), and SMILES are not canonicalised. Both missed matches
and false scientific links are possible: shared components, predictions, zero
labels or header misclassification can match without the same full formulation.
These counts are not a lower bound on confirmed scientific connections. See
review/cross_scale_candidate_interpretation.csv for case-level interpretation.

The scan's recall is bounded by its file formats, row/byte caps and keyword list,
so repositories it cannot place are reported separately rather than counted as
tools that publish nothing, and every distinct unplaced header is written out
for inspection. A zero is therefore a bounded non-detection, not proof of
absence.

Outputs (results/cross_scale_gap/):
    table_scale_coverage.csv   one row per repository
    table_candidate_overlap.csv   one row per repository pair that shares any candidate
    table_scale_columns.csv    every column that contributed, so the rules are auditable
    table_unattributed_columns.csv   every distinct column the rules could not place
    SUMMARY.md                 the headline counts

Usage
-----
    python scripts/20_measure_cross_scale_gap.py [--repos-cache DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import csv
import os
import re
import sys
sys.dont_write_bytecode = True
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CENSUS = ROOT / "data" / "census"
OUTDIR = ROOT / "results" / "cross_scale_gap"

#: Fall back to the environment variable the dataset loaders already use.
DEFAULT_CACHE = os.environ.get("BATTAGENT_REPOS_CACHE", "")

#: Read at most this much of any one file. The largest released tables are tens
#: of megabytes of featurised rows; the identifiers repeat long before that.
MAX_BYTES = 4 << 20
MAX_ROWS = 20000

SCALES = ("material", "interface", "cell")

#: Tracked alongside SCALES, not as a fourth scale: the subset of the cell-level
#: candidates whose cell-level evidence comes from a cycled cell.
CYCLED = "cell_cycled"
BUCKETS = SCALES + (CYCLED,)

# Checked before the scale rules. These columns exist in quantity and would
# otherwise dominate: model bookkeeping is not a measurement of anything, and a
# featurisation block or a formula weight is a descriptor of the candidate
# rather than a property that a downstream stage could gate on.
# Joined with "|" and compiled without re.VERBOSE on purpose: verbose mode
# strips literal spaces, which would silently break every multi-word rule.
EXCLUDE_RULES: tuple[str, ...] = (
    r"^(pred|true|target|label|actual|y (pred|true|test|train))\b",
    r"\b(pred|true|actual|prediction|predicted)$",
    r"^(prediction|predicted)\b",
    r"\b(train|test|valid|epoch|loss|mae|rmse|mse|r2|score|fold|seed|split)\b",
    r"^magpiedata\b",
    r"^(mean|std|min|max|range|mode|avg dev)\b",
    r"molecular weight|\bmol ?wt\b|exactmolwt|formula weight",
    r"num atoms|atom count|componentcount|\becfp\b|\bpca \d",
    r"^(index|id|idx|unnamed)\b",
)

# Ordered: the first pattern that matches a column name wins, so compound terms
# ("energy density" is a cell metric) are tested before their components.
# Subjects are normalised to space-separated lower-case tokens, so \b behaves.
SCALE_RULES: list[tuple[str, str]] = [
    # ---- cell level: something was assembled and cycled ----
    ("cell", r"coulombic|capacity retention|\bretention\b"),
    ("cell", r"energy density|power density|areal capac|specific capac|gravimetric capac"),
    ("cell", r"capac|\bcycl|discharge|\bc rate\b|\bcrate\b|rate capab"),
    ("cell", r"\bsoh\b|state of health|\beol\b|cycle life|calendar life"),
    ("cell", r"full cell|half cell|coin cell|pouch"),
    # ---- interface level: two phases and what happens between them ----
    ("interface", r"\bsei\b|\bcei\b|interfac|interphase|passivat"),
    ("interface", r"adsorpt|adsorb|\bbinding\b|\bbe\b|surface energy|work function"),
    ("interface", r"adhesion|work of separation|\be ad\b|\be bd\b|\be it\b"),
    ("interface", r"wetting|contact angle|dendrit|plating|stripping"),
    ("interface", r"impedance|\beis\b|charge transfer|\br ?ct\b"),
    # ---- material level: a property of one substance ----
    ("material", r"formation energy|\be form\b|\bform e\b|e hull|above hull|ehull"),
    ("material", r"band ?gap|\bgap\b|\bhomo\b|\blumo\b|electron affinity|ionization"),
    ("material", r"conductiv|\bcond\b|\bionic\b|diffusiv|diffusion coef"),
    ("material", r"migration barrier|activation energy|^ea$|\bbarrier\b"),
    ("material", r"redox|oxidation potential|reduction potential|\bvoltage\b"),
    ("material", r"stability window|upper bound|lower bound|total range"),
    ("material", r"bulk modulus|shear modulus|elastic|youngs|poisson|hardness|\bvoigt\b|\breuss\b"),
    ("material", r"melting|boiling|viscosit|dielectric|permittivity|solubilit"),
    ("material", r"\bdensity\b|\blogp\b|donor number|\bdn\b|transference"),
    ("material", r"glass transition|\btg\b|decomposition|thermal stab"),
]

#: Within the cell level, separates a cycled cell from a capacity that was
#: computed from the same DFT record as the voltage. Materials Project electrode
#: tables carry ``average_voltage`` beside ``capacity_grav``; both come from one
#: calculation, so pairing them is not a link across scales even though the
#: keyword rules place them on either side of the boundary. Only the patterns
#: below imply that something was actually assembled and cycled.
CYCLING_RULES: tuple[str, ...] = (
    r"coulombic|retention|\bcycl|\bsoh\b|state of health|\beol\b",
    r"\bc rate\b|\bcrate\b|rate capab|calendar|fade",
    r"(discharge|charge|measured|norm|normalised|normalized|observed)( \w+)? capac",
    r"full cell|half cell|coin cell|pouch",
)

#: Units are more reliable than names. Applied to the bracketed annotation of a
#: column ("k [mS/cm]") when no keyword rule fires.
UNIT_RULES: list[tuple[str, str]] = [
    ("cell", r"^m?ah\s*/\s*(g|cm|l)|^w?h\s*/\s*(kg|l|g)"),
    ("material", r"^[munµμ]?s\s*/\s*cm$|^s\s*/\s*m$|^cm2?\s*/\s*s$"),
    # Energy per unit area is an interface or surface energy whatever it is
    # called. This is what rescues LaTeX-formatted headers such as
    # ``$E_{it}$ $(J/m^2)$``, where the name carries no readable keyword.
    ("interface", r"^m?j\s*/\s*m\s*\^?\s*2$|^ev\s*/\s*(a|å|ang)\w*\s*\^?\s*2$"),
]

IDENTIFIER_RULES: list[tuple[str, str]] = [
    ("smiles", r"^(.* )?smiles?( .*)?$|^(.* )?smi$|\bsm$"),
    ("inchi", r"inchi"),
    ("mp_id", r"^(mp ?id|material ?id|materials ?id|mpid)$"),
    ("formula", r"formula|composition|^comp$|chemsys"),
    ("cas", r"^cas( number| rn)?$"),
]

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")
_FORMULA_OK = re.compile(r"^(?:[A-Z][a-z]?\d*\.?\d*)+$")
_SMILES_HINT = re.compile(r"[=#@\[\]()]|[cnos]")

#: A whole cell that is plausibly a structure string: no whitespace, no
#: underscores, and at least one of the characters that only appear in SMILES.
#: Deliberately strict, so that headers like ``k [mS/cm]`` (has a space) and
#: ``norm_capacity_3`` (has underscores) are not mistaken for data.
_SMILES_CELL = re.compile(r"^(?=.{8,})(?=.*[=#@\[])[A-Za-z0-9@+\-\[\]()=#$%/\\.]+$")

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".ipynb_checkpoints", "site-packages"}


_EXCLUDE = re.compile("|".join(EXCLUDE_RULES))
_CYCLING = re.compile("|".join(CYCLING_RULES))
_BRACKETED = re.compile(r"[\(\[]([^\)\]]{1,20})[\)\]]")


def normalise_column(name: str) -> str:
    """Lower-case and reduce every separator to a single space.

    Released tables write the same quantity as ``log(cond) (uS/cm)``,
    ``ionic_conductivity`` and ``Gravimetric Capacticy(mAh/g)``. Reducing to
    space-separated tokens lets one keyword list cover all three spellings and
    makes ``\\b`` behave (underscores are word characters, spaces are not).
    """
    return re.sub(r"[^a-z0-9]+", " ", name.strip().lower()).strip()


def classify_column(name: str) -> str | None:
    key = normalise_column(name)
    if not key or _EXCLUDE.search(key):
        return None
    if classify_identifier(name):
        # A column that names the candidate is not a property of it.
        # ``formula_discharge`` is the discharged-electrode formula, not a
        # discharge measurement, and must not be read as a cell-level result.
        return None
    for scale, pattern in SCALE_RULES:
        if re.search(pattern, key):
            return scale
    for annotation in _BRACKETED.findall(name.lower()):
        unit = annotation.strip()
        for scale, pattern in UNIT_RULES:
            if re.search(pattern, unit):
                return scale
    return None


def is_cycling_measurement(name: str) -> bool:
    return bool(_CYCLING.search(normalise_column(name)))


def classify_identifier(name: str) -> str | None:
    key = normalise_column(name)
    for kind, pattern in IDENTIFIER_RULES:
        if re.search(pattern, key):
            return kind
    return None


def normalise_formula(text: str) -> str | None:
    """Reduce a formula to sorted element:count so Li2Zn2S1 and Li2Zn2S agree."""
    value = text.strip().replace(" ", "")
    if not value or not _FORMULA_OK.match(value):
        return None
    counts: dict[str, float] = defaultdict(float)
    for element, number in _FORMULA_TOKEN.findall(value):
        counts[element] += float(number) if number else 1.0
    if not counts:
        return None
    smallest = min(counts.values())
    if smallest <= 0:
        return None
    reduced = {el: round(n / smallest, 3) for el, n in counts.items()}
    return " ".join(f"{el}{reduced[el]:g}" for el in sorted(reduced))


def canonical_identifier(kind: str, raw: str) -> str | None:
    value = (raw or "").strip()
    if not value or len(value) > 300:
        return None
    if kind == "formula":
        return normalise_formula(value)
    if kind == "smiles":
        # No RDKit dependency here, so this is an exact-string match and the
        # overlap it produces is a lower bound. Reject obvious non-SMILES cells.
        return value if len(value) >= 2 and _SMILES_HINT.search(value) else None
    if kind == "mp_id":
        return value.lower() if value.lower().startswith("mp-") else None
    return value.lower()


def _looks_like_data(header: list[str]) -> bool:
    """True when row 1 is data, not column names.

    A sizeable minority of released tables ship without a header. Treating
    their first data row as column names produces nonsense classifications, so
    such files are skipped and counted separately. The two tells are a row that
    is mostly numbers and a row containing a structure string; both have to be
    recognised without rejecting legitimate headers such as ``k [mS/cm]`` or
    ``norm_capacity_3``.
    """
    if not header:
        return True
    numeric = 0
    for cell in header:
        text = cell.strip()
        if len(text) > 80 or _SMILES_CELL.match(text):
            return True
        try:
            float(text)
        except ValueError:
            continue
        numeric += 1
    return numeric >= max(1, len(header) // 2)


def sniff_delimiter(line: str) -> str:
    counts = {d: line.count(d) for d in (",", "\t", ";", "|")}
    best = max(counts, key=lambda d: counts[d])
    return best if counts[best] else ","


def scan_csv(path: Path) -> dict:
    """Classify one table: which scales it reports, and for which candidates."""
    empty = {
        "scales": {}, "used": [], "unattributed": [], "headerless": False,
        "read": False, "truncated": False,
    }
    try:
        handle = open(path, encoding="utf-8-sig", newline="", errors="replace")
    except OSError:
        return empty
    with handle:
        first = handle.readline()
        if not first.strip():
            return empty
        handle.seek(0)
        try:
            reader = csv.reader(handle, delimiter=sniff_delimiter(first))
            header = next(reader, None)
        except (csv.Error, UnicodeDecodeError):
            return empty
        if not header or len(header) > 4000:
            return empty
        if _looks_like_data(header):
            return {**empty, "headerless": True}

        id_cols = [(i, name, classify_identifier(name)) for i, name in enumerate(header)]
        id_cols = [(i, name, kind) for i, name, kind in id_cols if kind]
        scale_cols = [(i, name, classify_column(name)) for i, name in enumerate(header)]
        scale_cols = [(i, name, scale) for i, name, scale in scale_cols if scale]

        # Columns the rules could not place. Reported so the scan's recall can
        # be checked rather than assumed.
        placed = {i for i, _, _ in scale_cols} | {i for i, _, _ in id_cols}
        unattributed = [
            name.strip()[:60]
            for i, name in enumerate(header)
            if i not in placed and name.strip() and not _EXCLUDE.search(normalise_column(name))
        ]
        if not scale_cols:
            return {**empty, "unattributed": unattributed, "read": True}

        # A cell-level column additionally lands in CYCLED when its name shows
        # the cell was actually cycled, so a theoretical capacity can be told
        # apart from a measured one downstream.
        buckets = [
            (i, CYCLED if scale == "cell" and is_cycling_measurement(name) else scale)
            for i, name, scale in scale_cols
        ]
        used = [
            (name, scale, int(scale == "cell" and is_cycling_measurement(name)))
            for _, name, scale in scale_cols
        ]
        keys = {scale for _, scale in buckets} | {scale for _, _, scale in scale_cols}
        if not id_cols:
            # The table reports properties but names no candidate: it counts
            # towards scale coverage, never towards a per-candidate join.
            return {
                "scales": {scale: set() for scale in keys},
                "used": used,
                "unattributed": unattributed,
                "headerless": False,
                "read": True,
                "truncated": False,
            }

        found: dict[str, set[str]] = {scale: set() for scale in keys}
        read = 0
        truncated = False
        for rows_seen, row in enumerate(reader):
            if rows_seen >= MAX_ROWS or read > MAX_BYTES:
                truncated = True
                break
            read += sum(len(cell) for cell in row) + len(row)
            ids = set()
            for i, _, kind in id_cols:
                if i < len(row):
                    canonical = canonical_identifier(kind, row[i])
                    if canonical:
                        ids.add(f"{kind}:{canonical}")
            if not ids:
                continue
            for (i, bucket), (_, _, scale) in zip(buckets, scale_cols):
                if i < len(row) and row[i].strip():
                    found[bucket].update(ids)
                    found[scale].update(ids)
        return {
            "scales": found,
            "used": used,
            "unattributed": unattributed,
            "headerless": False,
            "read": True,
            "truncated": truncated,
        }


def scan_repo(repo: Path) -> dict:
    per_scale: dict[str, set[str]] = {s: set() for s in BUCKETS}
    scales_any: set[str] = set()
    columns: list[tuple[str, str, str]] = []
    unattributed: Counter[str] = Counter()
    n_csv = n_headerless = n_with_scale = n_truncated = 0
    for dirpath, dirnames, filenames in os.walk(repo):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for filename in filenames:
            if not filename.lower().endswith((".csv", ".tsv")):
                continue
            n_csv += 1
            path = Path(dirpath) / filename
            result = scan_csv(path)
            n_headerless += int(result["headerless"])
            n_truncated += int(result["truncated"])
            unattributed.update(result["unattributed"])
            if not result["scales"]:
                continue
            n_with_scale += 1
            for scale, ids in result["scales"].items():
                if scale in SCALES:
                    scales_any.add(scale)
                per_scale[scale].update(ids)
            rel = str(path.relative_to(repo)).replace("\\", "/")
            columns.extend((rel, name, scale, cycled) for name, scale, cycled in result["used"])

    multi: dict[str, int] = {}
    for a, b in combinations(SCALES, 2):
        shared = per_scale[a] & per_scale[b]
        if shared:
            multi[f"{a}+{b}"] = len(shared)
    all_three = per_scale["material"] & per_scale["interface"] & per_scale["cell"]

    return {
        "n_csv": n_csv,
        "n_headerless": n_headerless,
        "n_with_scale": n_with_scale,
        "n_truncated": n_truncated,
        "scales": scales_any,
        "identifiers": {s: per_scale[s] for s in BUCKETS},
        "multi_scale_pairs": multi,
        "n_all_three": len(all_three),
        "n_material_and_cycled_cell": len(per_scale["material"] & per_scale[CYCLED]),
        "n_cycled_cell": len(per_scale[CYCLED]),
        "columns": columns,
        "unattributed": unattributed,
    }


def run():
    """Retain the original matching rules; replace the historical cohort/I/O."""
    from score_resources import read_csv, write_csv, snapshot_path, normalize_manifest, truth, manifest_eligible
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'data/resource_manifest.csv')
    args=parser.parse_args()
    rows=[normalize_manifest(r) for r in read_csv(args.manifest)]
    primaries=[r for r in rows if truth(r.get('is_primary')) and r.get('resource_kind','').lower() in {'code','both','code/data','code_and_data'}]
    score_file=ROOT/'data/scoring_resources.csv'
    score_status={r['resource_id']:r.get('scoring_status','') for r in read_csv(score_file)} if score_file.exists() else {}
    groups={}
    for r in primaries:
        groups.setdefault(r['resource_id'],[]).append(r)
    coverage=[]; columns=[]; unattributed=[]; identifiers={}; errors=[];candidate_rows=[]; table_reads=[];partial_results=[];encoding_notes=[]
    for rid,rels in sorted(groups.items()):
        row=rels[0]; meta={'resource_id':rid,'paper_ids':';'.join(sorted({r['paper_id'] for r in rels})),'paper_count':len({r['paper_id'] for r in rels}),'snapshot_path':row['snapshot_path'],'source_url':row['source_url']}
        paths={r['snapshot_path'] for r in rels}
        if len(paths)>1:
            errors.append({'resource_id':rid,'error':'Same resource ID maps to multiple snapshots; choose one frozen snapshot before deduplicating.'})
            coverage.append({**meta,'scan_status':'inconsistent_snapshot'});continue
        if not manifest_eligible(row):
            coverage.append({**meta,'scan_status':'not_scanned_unverified_or_unattributed'});continue
        if score_status.get(rid)!='scored':
            coverage.append({**meta,'scan_status':'not_scanned_no_completed_source_score'});continue
        try:
            repo=snapshot_path(row)
            if not repo.is_dir():raise ValueError('Missing snapshot directory')
            for p in repo.rglob('*'):
                if p.is_symlink():raise ValueError('Symlink forbidden in static scan: '+str(p.relative_to(repo)))
            result=scan_repo(repo)
            # Independently expose per-table read failures rather than calling them zero.
            read_failures=0
            for directory,dirs,filenames in os.walk(repo):
                dirs[:]=[d for d in dirs if d not in SKIP_DIRS]
                for filename in filenames:
                    if not filename.lower().endswith(('.csv','.tsv')):continue
                    p=Path(directory)/filename; t=scan_csv(p)
                    decode_bad=False
                    with p.open('rb') as header_stream:header_bytes=header_stream.readline(MAX_BYTES)
                    try:header_bytes.decode('utf-8-sig',errors='strict')
                    except UnicodeDecodeError:
                        decode_bad=True
                        try:possible_header=header_bytes.decode('gb18030',errors='strict')[:3000]
                        except UnicodeDecodeError:possible_header='not_decodable_as_GB18030'
                        import hashlib
                        # CSV/Excel cannot safely display embedded NUL/control bytes.
                        # Escape them for presentation; original source bytes/hash remain evidence.
                        def printable_header(value):
                            return ''.join(c if c.isprintable() or c in '\t\r\n' else '\\u%04x'%ord(c) for c in value)
                        encoding_notes.append({'resource_id':rid,'file':p.relative_to(repo).as_posix(),'original_utf8_header':printable_header(header_bytes.decode('utf-8',errors='replace')[:3000]),'possible_GB18030_header_not_used_for_matching':printable_header(possible_header),'header_sha256':hashlib.sha256(header_bytes).hexdigest(),'header_hex_prefix_256bytes':header_bytes[:256].hex(),'header_bytes_length':len(header_bytes),'display_note':'Control bytes escaped as literal \\uXXXX; source bytes unchanged','metadata_type':'AppleDouble metadata; not a research CSV' if header_bytes.startswith(bytes.fromhex('00051607')) else 'encoding unresolved','decision':'header_encoding_unresolved; original match rules retained; exclude resource from complete-scan denominator'})
                    read_success=t['read'] and not decode_bad
                    headerless=t['headerless'] and not decode_bad
                    failed=not read_success and not headerless
                    read_failures+=int(failed)
                    table_reads.append({'resource_id':rid,'file':str(p.relative_to(repo)),'read_success':read_success,'headerless':headerless,'truncated':t['truncated'],'unread_or_empty':failed,'header_decode_error':decode_bad,'raw_rule_read_success':t['read'],'raw_rule_headerless':t['headerless']})
            pairs=result['multi_scale_pairs']
            values={
                'tables_found':result['n_csv'],'tables_without_header':result['n_headerless'],
                'tables_attributed_to_a_scale':result['n_with_scale'],'tables_truncated_by_scan_limit':result['n_truncated'],
                'tables_unread_or_empty':read_failures,
                'reports_material':int('material' in result['scales']),'reports_interface':int('interface' in result['scales']),
                'reports_cell':int('cell' in result['scales']),'scales_covered':len(result['scales']),
                'candidates_material':len(result['identifiers']['material']),'candidates_interface':len(result['identifiers']['interface']),
                'candidates_cell':len(result['identifiers']['cell']),'candidates_cell_cycled':result['n_cycled_cell'],
                'candidates_material_and_interface':pairs.get('material+interface',0),'candidates_material_and_cell':pairs.get('material+cell',0),
                'candidates_interface_and_cell':pairs.get('interface+cell',0),'candidates_material_and_cycled_cell':result['n_material_and_cycled_cell'],
                'candidates_all_three_scales':result['n_all_three'],'spans_two_scales_for_one_candidate':int(bool(pairs)),
            }
            status='scanned_attributed' if result['scales'] else 'scanned_unattributed'
            if read_failures:
                partial_results.append({**meta,'scan_status':'partial_table_read_lower_bounds_only',**values})
                status='not_fully_scanned_table_read_failure'
                for k in list(values):
                    if not k.startswith('tables_'):values[k]=''
            coverage.append({**meta,'scan_status':status,**values})
            if not read_failures:identifiers[rid]=result['identifiers']
            for rel,name,scale,cycled in result['columns']:
                columns.append({'resource_id':rid,'file':rel,'locator':rel+':1 (header)','column':name,'scale':scale,'is_cycling_measurement':cycled})
            for name,n in result['unattributed'].most_common():
                unattributed.append({'resource_id':rid,'column':name,'times_seen':n})
            for scale,candidates in result['identifiers'].items():
                for candidate in sorted(candidates):candidate_rows.append({'resource_id':rid,'scale':scale,'candidate':candidate})
        except Exception as exc:
            errors.append({'resource_id':rid,'error':str(exc)})
            coverage.append({**meta,'scan_status':'scan_failed'})
    flat={rid:set().union(*sets.values()) for rid,sets in identifiers.items()}; overlap=[]
    for (a,ids_a),(b,ids_b) in combinations(sorted(flat.items()),2):
        shared=ids_a&ids_b
        if not shared:continue
        cross=set()
        for sa,sb in combinations(SCALES,2):
            cross|=identifiers[a][sa]&identifiers[b][sb]
            cross|=identifiers[a][sb]&identifiers[b][sa]
        measured=(identifiers[a]['material']&identifiers[b][CYCLED])|(identifiers[b]['material']&identifiers[a][CYCLED])
        overlap.append({'resource_a':a,'resource_b':b,'candidates_a':len(ids_a),'candidates_b':len(ids_b),'shared_candidates':len(shared),'jaccard':round(len(shared)/len(ids_a|ids_b),5),'shared_across_different_scales':len(cross),'shared_material_to_cycled_cell':len(measured),'example':sorted(shared)[0][:80]})
    overlap.sort(key=lambda r:r['shared_candidates'],reverse=True)
    base=['resource_id','paper_ids','paper_count','snapshot_path','source_url','scan_status']
    metric=['tables_found','tables_without_header','tables_attributed_to_a_scale','tables_truncated_by_scan_limit','tables_unread_or_empty','reports_material','reports_interface','reports_cell','scales_covered','candidates_material','candidates_interface','candidates_cell','candidates_cell_cycled','candidates_material_and_interface','candidates_material_and_cell','candidates_interface_and_cell','candidates_material_and_cycled_cell','candidates_all_three_scales','spans_two_scales_for_one_candidate']
    write_csv(ROOT/'data/cross_scale_coverage.csv',coverage,base+metric)
    write_csv(ROOT/'data/cross_scale_partial_results.csv',partial_results,base+metric)
    write_csv(ROOT/'data/cross_scale_columns.csv',columns,['resource_id','file','locator','column','scale','is_cycling_measurement'])
    write_csv(ROOT/'data/cross_scale_unattributed.csv',unattributed,['resource_id','column','times_seen'])
    write_csv(ROOT/'data/cross_scale_table_reads.csv',table_reads,['resource_id','file','read_success','headerless','truncated','unread_or_empty','header_decode_error','raw_rule_read_success','raw_rule_headerless'])
    write_csv(ROOT/'data/cross_scale_encoding_notes.csv',encoding_notes,['resource_id','file','original_utf8_header','possible_GB18030_header_not_used_for_matching','header_sha256','decision','header_hex_prefix_256bytes','header_bytes_length','display_note','metadata_type'])
    write_csv(ROOT/'data/cross_scale_candidates.csv',candidate_rows,['resource_id','scale','candidate'])
    write_csv(ROOT/'data/cross_scale_overlap.csv',overlap,['resource_a','resource_b','candidates_a','candidates_b','shared_candidates','jaccard','shared_across_different_scales','shared_material_to_cycled_cell','example'])
    write_csv(ROOT/'data/cross_scale_errors.csv',errors,['resource_id','error'])
    attributed=[r for r in coverage if isinstance(r.get('scales_covered'),int) and r['scales_covered']>0]
    n=len(identifiers)
    summary={'unique_resources':n,'possible_pairs':n*(n-1)//2,'primary_paper_count':len(primaries),'unique_primary_resource_count':len(groups),'scanned_resource_count':n,'unscanned_resource_count':len(groups)-n,'attributed_resource_count':len(attributed),'unattributed_resource_count':sum(r.get('scales_covered')==0 for r in coverage),'possible_scanned_resource_pairs':n*(n-1)//2,'pairs_sharing_candidates':len(overlap),'pairs_cross_scale':sum(r['shared_across_different_scales']>0 for r in overlap),'pairs_material_to_cycled_cell':sum(r['shared_material_to_cycled_cell']>0 for r in overlap),'resources_candidate_two_scales':sum(r.get('spans_two_scales_for_one_candidate',0)>0 for r in attributed),'resources_material_to_cycled_cell':sum(r.get('candidates_material_and_cycled_cell',0)>0 for r in attributed),'resources_all_three_scales':sum(r.get('candidates_all_three_scales',0)>0 for r in attributed),'scale_coverage_counts':{str(k):sum(r['scales_covered']==k for r in attributed) for k in (1,2,3)},'tables_found':sum(r.get('tables_found',0) for r in coverage),'tables_unread_or_empty':sum(r.get('tables_unread_or_empty',0) for r in coverage),'tables_truncated':sum(r.get('tables_truncated_by_scan_limit',0) for r in coverage),'max_rows':MAX_ROWS,'max_bytes_constant':MAX_BYTES,'partial_table_read_resource_count':len(partial_results),'errors':len(errors),'limitation':'Original CSV/TSV matching and bounded prefix only; absence is bounded non-detection. Original byte counter estimates decoded character lengths, not raw UTF-8 bytes; retained verbatim for comparability.'}
    (ROOT/'data/cross_scale_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False));return 1 if errors else 0


if __name__=='__main__':raise SystemExit(run())
