#!/usr/bin/env python3
"""
RRL/CRL Objective Scoring Script — static (local-clone)
=====================================================
Reads census_opensource_full.csv, clones repos locally (shallow),
scans ALL files, and computes binary indicators R1-R7 / D1,D2,D3
for each code=yes|partial paper.

static changes vs prior:
  - Reframes CRL as workflow-adaptation-cost proxy
  - Keeps CSV columns D1/D2/D3 stable, but updates semantics:
      D1: parameter/task configurability
      D2: data interoperability
      D3: interface invocability
  - Adds notebook code-cell, MATLAB script, standard structure/molecule file,
    domain table-header, and pymatgen-MSON evidence
  - Tightens D3 so README-only dependency mentions and bare main.py do not count,
    while accepting executable top-level notebooks/MATLAB scripts as light
    workflow-entry evidence

prior changes vs static:
  - Replaces GitHub API sampling with `git clone --depth 1` + full scan
  - Adds Zenodo download support (REST API → unzip)
  - Eliminates rate-limit issues (no GitHub API calls)
  - Scans ALL .py and dependency files (no 2-file sampling)
  - Fixes false negatives (e.g. LiSSE-MP D2 missed due to sampling)

Scoring formula unchanged from static:
  - RRL: additive, core R2+R3+R4+R6 = 0-4, bonus R5+R7 → cap 5
  - CRL: D1+D2+D3 (0-3), using static D1/D2/D3 semantics

Output: tool_scores_auto.csv

Usage:
    python scripts/score_rrl_crl.py              # full run
    python scripts/score_rrl_crl.py --resume      # skip already-scored repos
    python scripts/score_rrl_crl.py --dry-run     # show repos & source resolution
    python scripts/score_rrl_crl.py --clean        # delete cache after scoring
    python scripts/score_rrl_crl.py --timeout 180  # clone timeout in seconds
"""

from __future__ import annotations
import csv, json, os, sys, re, pathlib, argparse, subprocess, shutil, zipfile, tarfile
import xml.etree.ElementTree as ET
from typing import Optional, List, Tuple, Dict
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VAULT = pathlib.Path(__file__).resolve().parent.parent
CSV_IN = VAULT / "census_data" / "census_opensource_full.csv"
CSV_OUT = VAULT / "data" / "tool_scores_auto.csv"
CACHE_DIR = VAULT / "_repos_cache"
OVERRIDES_CSV = VAULT / "data" / "manual_overrides.csv"
MAX_PY_BYTES = 20 * 1024 * 1024   # 20 MB total .py text cap
MAX_SINGLE_FILE = 2 * 1024 * 1024  # 2 MB per file cap

# ---------------------------------------------------------------------------
# Binary extension filter
# ---------------------------------------------------------------------------
BINARY_EXTS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".pt", ".pth", ".ckpt", ".h5", ".pkl", ".safetensors", ".bin", ".pb",
    ".npy", ".npz", ".hdf5", ".parquet",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".so", ".dylib", ".dll", ".o", ".a",
    ".whl", ".egg",
    ".pdf", ".doc", ".docx", ".pptx",
    ".mp4", ".avi", ".mov", ".wav", ".mp3",
    ".pyc", ".pyo",
})

STANDARD_CHEM_EXTS = frozenset({
    ".cif", ".xyz", ".extxyz", ".sdf", ".mol", ".mol2", ".pdb",
    ".poscar", ".vasp", ".traj", ".xsf", ".lmp",
})
STANDARD_CHEM_NAMES = frozenset({"POSCAR", "CONTCAR"})
STANDARD_CHEM_DIRS = frozenset({
    "data", "dataset", "datasets", "input", "inputs", "example",
    "examples", "reference", "references", "structure", "structures",
    "scripts", "notebooks",
})
MAX_NOTEBOOK_BYTES = 15 * 1024 * 1024
MAX_JSON_EVIDENCE_BYTES = 3 * 1024 * 1024
MAX_NOTEBOOK_TEXT_BYTES = 20 * 1024 * 1024
MAX_JSON_TEXT_BYTES = 10 * 1024 * 1024
MAX_CHEM_EVIDENCE_FILES = 50
MAX_MSON_EVIDENCE_FILES = 50
MAX_MATLAB_BYTES = 10 * 1024 * 1024
MAX_TABLE_HEADER_BYTES = 2 * 1024 * 1024
MAX_TABLE_HEADER_FILES = 200
TABLE_EXTS = frozenset({".csv", ".tsv", ".xlsx"})
CACHE_DIR_NAMES = frozenset({
    ".git", ".hg", ".svn", ".ipynb_checkpoints", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".nox",
    ".venv", "venv", "env", "node_modules", "dist", "build",
})


def _is_binary_ext(fname: str) -> bool:
    """Quick check: skip known binary extensions."""
    return pathlib.PurePosixPath(fname).suffix.lower() in BINARY_EXTS


def _is_hidden_or_cache_part(part: str) -> bool:
    """True for hidden/cache path components that should not create evidence."""
    lower = part.lower()
    if lower == ".devcontainer":
        return False
    if lower in CACHE_DIR_NAMES:
        return True
    return part.startswith(".")


def _is_pruned_dir(part: str) -> bool:
    """Directories that are safe to skip entirely during repository walks."""
    lower = part.lower()
    if lower == ".devcontainer":
        return False
    return lower in CACHE_DIR_NAMES or part.startswith(".")


def _is_ignored_relpath(rel: str) -> bool:
    """Skip hidden/checkpoint/cache paths for scanning evidence."""
    return any(_is_hidden_or_cache_part(part) for part in pathlib.PurePosixPath(rel).parts)


def _is_top_or_first_level(rel: str) -> bool:
    """Root-level or one directory below root."""
    return len(pathlib.PurePosixPath(rel).parts) <= 2


# ---------------------------------------------------------------------------
# Local clone / download helpers
# ---------------------------------------------------------------------------
def _has_payload_files(path: pathlib.Path) -> bool:
    """Return True when a cached source has files outside VCS metadata.

    A failed/interrupted git clone can leave a non-empty directory containing
    only `.git/`. Treating that as a cache hit produces an R1=True row with
    zero scannable files, so cache reuse must check the worktree payload.
    """
    if not path.exists():
        return False
    for item in path.rglob("*"):
        if item.is_file() and ".git" not in item.relative_to(path).parts:
            return True
    return False


def clone_repo(url: str, dest_dir: pathlib.Path, timeout: int = 120, retries: int = 3) -> bool:
    """Shallow-clone a git repo. Skips if dest_dir already has content.
    Retries on transient network errors (early EOF, timeout)."""
    if _has_payload_files(dest_dir):
        return True  # already cached
    if dest_dir.exists():
        shutil.rmtree(dest_dir, ignore_errors=True)
    for attempt in range(1, retries + 1):
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "--single-branch", "--quiet", url, str(dest_dir)],
                timeout=timeout, check=True,
                capture_output=True,
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            shutil.rmtree(dest_dir, ignore_errors=True)
            if attempt < retries:
                import time
                wait = 5 * attempt
                print(f"    clone attempt {attempt}/{retries} failed, retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    clone failed after {retries} attempts: {e}")
                return False


MAX_ZENODO_BYTES = 200 * 1024 * 1024  # 200 MB — skip huge data-only deposits

# Source corrections required to make full rebuilds reproducible from the
# census table. Some original data-availability statements point to a GitHub
# profile or omit a cloneable URL; these entries were resolved during manual
# review and are kept here so a fresh run does not regress to R1=False.
SOURCE_OVERRIDES = {
    "10208": ("github", "https://github.com/mejiadongs/Solid-Polymer-Electrolytes-in-Lithium-Metal-Batteries.git",
              "gh_mejiadongs__Solid-Polymer-Electrolytes-in-Lithium-Metal-Batteries"),
    "10245": ("github", "https://github.com/MohamadA-RJ/OptiANN_BatteryCapPred.git",
              "gh_MohamadA-RJ__OptiANN_BatteryCapPred"),
    "10383": ("github", "https://github.com/damdaepark/pymatcc.git",
              "gh_damdaepark__pymatcc"),
    "40150": ("github", "https://github.com/EgemonyeThankGod200/ThankGod_C_Egemonye_and_Tomsmith_O_Unimuke_Ge12C12_and_Si12C12_Nanostructure_Battery.git",
              "gh_EgemonyeThankGod200__Ge12C12_Si12C12"),
    "20180": ("other",
              "https://gitlab.brindisi.enea.it/claudio.ronchetti/ai4mat",
              "other_20180"),
    "10797": ("github",
              "https://github.com/BattModels/Clio-LabViewControl.git",
              "gh_BattModels__Clio-LabViewControl"),
    "20229": ("github",
              "https://github.com/dftfeDevelopers/dftfe.git",
              "gh_dftfeDevelopers__dftfe"),
}

# Entries that must remain unavailable even if an unrelated or partial local
# cache exists. 10581 was previously mapped to an unrelated LLZO Deep Potential
# master's-project repository and must not be scored from that code.
FORCE_UNAVAILABLE = {
    "10581": "Clone/download failed: https://github.com/kapil-pareek/LLZO_01.git",
    "20362": "Excluded: code available on request only, not publicly accessible",
    "10031": "Excluded: generic tool citation only (deepmd-kit), not paper-specific code",
    "10135": "Excluded: Zenodo archive contains dataset only, no code scripts",
}


def download_zenodo(url: str, dest_dir: pathlib.Path) -> bool:
    """Download files from a Zenodo record into dest_dir.
    Supports both https://zenodo.org/records/XXXXX and doi:10.5281/zenodo.XXXXX.
    Skips records whose total size exceeds MAX_ZENODO_BYTES."""
    if _has_payload_files(dest_dir):
        return True  # already cached
    if dest_dir.exists():
        shutil.rmtree(dest_dir, ignore_errors=True)

    # Extract record ID
    record_id = None
    m = re.search(r'zenodo\.org/records?/(\d+)', url)
    if m:
        record_id = m.group(1)
    else:
        m = re.search(r'zenodo\.(\d+)', url)
        if m:
            record_id = m.group(1)
    if not record_id:
        print(f"    Cannot parse Zenodo record ID from: {url}")
        return False

    # Fetch file list via REST API
    api_url = f"https://zenodo.org/api/records/{record_id}"
    try:
        req = Request(api_url, headers={"Accept": "application/json", "User-Agent": "rrl-crl-scorer/1.0"})
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except (HTTPError, URLError, TimeoutError) as e:
        print(f"    Zenodo API error: {e}")
        return False

    files_list = data.get("files", [])
    if not files_list:
        print(f"    No files in Zenodo record {record_id}")
        return False

    total_size = sum(f.get("size", 0) for f in files_list)
    if total_size > MAX_ZENODO_BYTES:
        print(f"    Zenodo record too large ({total_size/1024/1024:.0f} MB > {MAX_ZENODO_BYTES/1024/1024:.0f} MB limit), skipping")
        return False

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Try to find an archive first
    archive = None
    for finfo in files_list:
        fname = finfo.get("key", "")
        if fname.endswith((".zip", ".tar.gz", ".tgz")):
            archive = finfo
            break

    if archive:
        dl_url = archive["links"]["self"]
        fname = archive["key"]
        local_path = dest_dir / fname
        try:
            req = Request(dl_url, headers={"User-Agent": "rrl-crl-scorer/1.0"})
            with urlopen(req, timeout=300) as resp:
                local_path.write_bytes(resp.read())
            # Extract
            if fname.endswith(".zip"):
                with zipfile.ZipFile(local_path) as zf:
                    zf.extractall(dest_dir)
            elif fname.endswith((".tar.gz", ".tgz")):
                with tarfile.open(local_path) as tf:
                    tf.extractall(dest_dir)
            local_path.unlink(missing_ok=True)
            return True
        except Exception as e:
            print(f"    Zenodo download/extract failed: {e}")
            shutil.rmtree(dest_dir, ignore_errors=True)
            return False
    else:
        # Download individual files (max 20)
        count = 0
        for finfo in files_list[:20]:
            dl_url = finfo["links"]["self"]
            fname = finfo["key"]
            local_path = dest_dir / fname
            try:
                req = Request(dl_url, headers={"User-Agent": "rrl-crl-scorer/1.0"})
                with urlopen(req, timeout=120) as resp:
                    local_path.write_bytes(resp.read())
                count += 1
            except Exception:
                pass
        return count > 0


def _extract_notebook_code(path: pathlib.Path) -> str:
    """Return concatenated code cells from a Jupyter notebook."""
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return ""
    parts: List[str] = []
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            parts.append("".join(src))
        elif isinstance(src, str):
            parts.append(src)
    return "\n".join(parts)


def _is_standard_chem_file(rel: str) -> bool:
    """Detect explicit chemistry/structure interchange files, not generic data."""
    p = pathlib.PurePosixPath(rel)
    if p.name.upper() in STANDARD_CHEM_NAMES:
        return True
    if p.suffix.lower() not in STANDARD_CHEM_EXTS:
        return False
    # Root-level standard chemistry files are explicit enough. For nested files,
    # require a data/example/reference-like path to avoid accidental matches.
    if len(p.parts) == 1:
        return True
    return any(part.lower() in STANDARD_CHEM_DIRS for part in p.parts[:-1])


def _json_has_pymatgen_mson(path: pathlib.Path) -> bool:
    """True for structured pymatgen/ASE-style JSON metadata, false for text corpora."""
    try:
        prefix = path.read_text(encoding="utf-8", errors="replace")[:MAX_JSON_EVIDENCE_BYTES]
    except Exception:
        return False
    lower = prefix.lower()
    if '"given_text"' in lower or '"result"' in lower or '"abstract"' in lower:
        return False
    has_module = '"@module"' in lower or '\\"@module\\"' in lower
    return has_module and ("pymatgen" in lower or "ase" in lower)


def _read_delimited_header(path: pathlib.Path, delimiter: Optional[str] = None) -> List[str]:
    """Read the first non-empty header row from CSV/TSV with stdlib only."""
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            sample = f.read(8192)
            if not sample.strip():
                return []
            f.seek(0)
            if delimiter is None:
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
                except csv.Error:
                    dialect = csv.excel
            else:
                dialect = csv.excel_tab if delimiter == "\t" else csv.excel
                dialect.delimiter = delimiter
            reader = csv.reader(f, dialect)
            for row in reader:
                fields = [c.strip() for c in row if c and c.strip()]
                if fields:
                    fields = fields[:80]
                    return fields if _looks_like_table_header(fields) else []
    except Exception:
        return []
    return []


def _xlsx_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    """Extract sharedStrings from an XLSX archive."""
    try:
        data = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return []
    values: List[str] = []
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    for si in root.findall(f".//{ns}si"):
        texts = [t.text or "" for t in si.findall(f".//{ns}t")]
        values.append("".join(texts))
    return values


def _read_xlsx_header(path: pathlib.Path) -> List[str]:
    """Read the first non-empty row from the first XLSX worksheet."""
    try:
        with zipfile.ZipFile(path) as zf:
            shared = _xlsx_shared_strings(zf)
            sheet_names = sorted(
                name for name in zf.namelist()
                if re.match(r"xl/worksheets/sheet\d+\.xml$", name)
            )
            if not sheet_names:
                return []
            root = ET.fromstring(zf.read(sheet_names[0]))
    except Exception:
        return []

    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    for row in root.findall(f".//{ns}sheetData/{ns}row"):
        fields: List[str] = []
        for cell in row.findall(f"{ns}c"):
            cell_type = cell.attrib.get("t", "")
            value = ""
            if cell_type == "s":
                v = cell.find(f"{ns}v")
                if v is not None and v.text and v.text.isdigit():
                    idx = int(v.text)
                    if 0 <= idx < len(shared):
                        value = shared[idx]
            elif cell_type == "inlineStr":
                value = "".join(t.text or "" for t in cell.findall(f".//{ns}t"))
            else:
                v = cell.find(f"{ns}v")
                if v is not None and v.text:
                    value = v.text
            if value.strip():
                fields.append(value.strip())
        if fields:
            fields = fields[:80]
            return fields if _looks_like_table_header(fields) else []
    return []


def _looks_like_table_header(fields: List[str]) -> bool:
    """Reject prose/data rows so paper corpora do not become D2 evidence."""
    if not fields:
        return False
    normalized = [f.strip() for f in fields if f and f.strip()]
    if not normalized:
        return False
    # Text-mining corpora often have no header; their first row is a relation
    # label plus a long sentence. Those should not be treated as field names.
    if any(len(f) > 80 for f in normalized):
        return False
    if any(re.search(r"<e\d+>|given_text|result|sentences?|relation", f, re.I) for f in normalized):
        return False
    return True


def _domain_header_matches(fields: List[str]) -> Tuple[bool, List[str]]:
    """Detect standard chemistry/materials or battery-performance table fields."""
    normalized = []
    for field in fields:
        text = re.sub(r"[_\-/]+", " ", field.lower()).strip()
        text = re.sub(r"\s+", " ", text)
        normalized.append(text)
    joined = " | ".join(normalized)

    strong_patterns = {
        "smiles": r"\bsmiles?\b",
        "inchi": r"\binchi(?:key)?\b",
        "formula": r"\bformula\b",
        "composition": r"\bcomposition\b",
        "structure": r"\bstructure\b",
        "cif": r"\bcif\b",
        "poscar": r"\bposcar\b",
        "material_id": r"\bmaterial id\b|\bmaterial_id\b|\bmaterials project id\b|\bmp id\b|\bmpid\b",
        "space group": r"\bspace group\b|\bspacegroup\b",
    }
    matches: List[str] = []
    for label, pat in strong_patterns.items():
        if re.search(pat, joined):
            matches.append(label)
    if matches:
        return True, matches

    battery_patterns = {
        "voltage": r"\bvoltage\b",
        "current": r"\bcurrent\b",
        "capacity": r"\bcapacity\b",
        "conductivity": r"\bconductivity\b",
        "ionic conductivity": r"\bionic conductivity\b",
        "diffusion": r"\bdiffusion\b",
        "cycle": r"\bcycle\b|\bcycles\b|\bcycle life\b",
        "soc": r"\bsoc\b|\bstate of charge\b",
        "soh": r"\bsoh\b|\bstate of health\b",
        "rul": r"\brul\b|\bremaining useful life\b",
        "energy density": r"\benergy density\b",
        "power density": r"\bpower density\b",
        "coulombic efficiency": r"\bcoulombic efficiency\b",
    }
    for label, pat in battery_patterns.items():
        if re.search(pat, joined):
            matches.append(label)
    return len(matches) >= 2, matches


def _read_table_header(path: pathlib.Path, rel: str) -> List[str]:
    """Read a table header for D2 evidence. Failures are intentionally silent."""
    suffix = pathlib.PurePosixPath(rel).suffix.lower()
    if suffix == ".csv":
        return _read_delimited_header(path)
    if suffix == ".tsv":
        return _read_delimited_header(path, "\t")
    if suffix == ".xlsx":
        return _read_xlsx_header(path)
    return []


def _code_has_external_input_loader(text: str) -> bool:
    """Detect code that loads replaceable external inputs via a path/variable."""
    if not text.strip():
        return False
    if re.search(
        r"\bload\s+[A-Za-z0-9_./\\ -]+\.(?:mat|csv|tsv|xlsx|xls|txt|dat)\b",
        text,
        re.I,
    ):
        return True
    loader_call = re.search(
        r"(?:pd\.)?read_(?:csv|excel|table)\s*\(|"
        r"np\.load\s*\(|loadmat\s*\(|scipy\.io\.loadmat\s*\(|"
        r"readtable\s*\(|readmatrix\s*\(|xlsread\s*\(|csvread\s*\(|"
        r"importdata\s*\(|\bload\s*\(",
        text,
        re.I,
    )
    if not loader_call:
        return False
    literal_path = re.search(
        r"['\"][^'\"]+\.(?:csv|tsv|xlsx|xls|txt|json|npy|npz|mat|dat)['\"]",
        text,
        re.I,
    )
    if literal_path:
        return True
    path_variable = re.search(
        r"\b(?:data|input|file|filename|filepath|path|datadir|data_dir)\w*\s*=\s*",
        text,
        re.I,
    )
    loader_uses_variable = re.search(
        r"(?:read_(?:csv|excel|table)|np\.load|loadmat|readtable|readmatrix|"
        r"xlsread|csvread|importdata|\bload)\s*\(\s*"
        r"(?:data|input|file|filename|filepath|path|datadir|data_dir)\w*",
        text,
        re.I,
    )
    return bool(path_variable and loader_uses_variable)


def _matlab_is_main_script(text: str, rel: str, readme: str = "") -> bool:
    """Accept root/first-level MATLAB scripts, but reject pure function helpers."""
    if not text.strip():
        return False
    name = pathlib.PurePosixPath(rel).name
    if readme and re.search(
        rf"(?:run|execute|open)\s+(?:the\s+)?`?{re.escape(name)}`?",
        readme,
        re.I,
    ):
        return True

    lines = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("%"):
            continue
        lines.append(stripped)
    if not lines:
        return False
    first = lines[0].lower()
    if first.startswith("function") or first.startswith("classdef"):
        return False

    helper_name = re.search(r"(?:helper|utils?|function|plot|figure|metric|loss)", name, re.I)
    if helper_name and not re.search(r"(?:main|run|demo|example|train|predict|script)", name, re.I):
        return False

    executable_patterns = [
        r"\breadtable\s*\(", r"\breadmatrix\s*\(", r"\bxlsread\s*\(",
        r"\bcsvread\s*\(", r"\bload\s*\(", r"\bimportdata\s*\(",
        r"\bfitr?\w*\s*\(", r"\btrain\w*\s*\(", r"\bpredict\s*\(",
        r"\bsave\s*\(", r"\bwritetable\s*\(", r"\bplot\s*\(",
        r"=", r"\bfor\b", r"\bif\b",
    ]
    body = "\n".join(lines[:200])
    return any(re.search(pat, body, re.I) for pat in executable_patterns)


def scan_local_repo(repo_dir: pathlib.Path) -> dict:
    """Walk a local directory and return structured content for scoring.
    Returns {files, tree_data, readme, dep_text, py_text, notebook_text,
    matlab_text, executable_notebooks, matlab_main_scripts, table_d2_headers,
    chem_files, mson_files}."""
    files: List[str] = []
    tree_data: List[dict] = []
    readme = ""
    dep_parts: List[str] = []
    py_parts: List[str] = []
    notebook_parts: List[str] = []
    matlab_parts: List[str] = []
    mson_parts: List[str] = []
    chem_files: List[str] = []
    mson_files: List[str] = []
    executable_notebooks: List[str] = []
    matlab_main_scripts: List[str] = []
    table_d2_headers: List[str] = []
    py_bytes_total = 0
    notebook_bytes_total = 0
    matlab_bytes_total = 0
    json_bytes_total = 0
    table_header_files = 0

    dep_names = {"requirements.txt", "environment.yml", "environment.yaml",
                 "setup.py", "setup.cfg", "pyproject.toml", "Pipfile",
                 "conda.yml", "conda.yaml", "Pipfile.lock", "poetry.lock",
                 "conda-lock.yml"}
    r6_size_exts = {".csv", ".json", ".xlsx", ".parquet", ".hdf5", ".npz", ".npy"}

    for dirpath, dirnames, filenames in os.walk(repo_dir):
        dirnames[:] = [d for d in dirnames if not _is_pruned_dir(d)]
        for fname in filenames:
            full = pathlib.Path(dirpath) / fname
            rel = os.path.relpath(full, repo_dir).replace(os.sep, "/")
            if _is_ignored_relpath(rel):
                continue
            name_lower = fname.lower()
            files.append(rel)
            if len(chem_files) < MAX_CHEM_EVIDENCE_FILES and _is_standard_chem_file(rel):
                chem_files.append(rel)

            is_dep_file = fname in dep_names or (
                ("requirement" in name_lower or "pip_env" in name_lower or "conda_env" in name_lower)
                and name_lower.endswith(".txt")
            )
            suffix = pathlib.PurePosixPath(rel).suffix.lower()
            needs_size = (
                name_lower.endswith((".ipynb", ".json", ".py", ".m", ".csv", ".tsv", ".xlsx"))
                or name_lower.startswith("readme")
                or is_dep_file
                or suffix in r6_size_exts
            )
            fsize: Optional[int] = None
            tree_item = {"path": rel, "type": "blob"}
            if needs_size:
                try:
                    fsize = full.stat().st_size
                    tree_item["size"] = fsize
                except OSError:
                    pass
            tree_data.append(tree_item)

            # Notebook code cells — markdown/output are ignored to avoid prose hits.
            if (
                name_lower.endswith(".ipynb")
                and fsize is not None
                and fsize <= MAX_NOTEBOOK_BYTES
                and notebook_bytes_total < MAX_NOTEBOOK_TEXT_BYTES
            ):
                content = _extract_notebook_code(full)
                notebook_bytes_total += len(content.encode("utf-8", errors="replace"))
                if content:
                    notebook_parts.append(f"\n# --- {rel} ---\n" + content)
                    if _is_top_or_first_level(rel):
                        executable_notebooks.append(rel)

            # MATLAB scripts are common in older battery-ML repositories. Treat
            # top-level executable scripts as light workflow-entry evidence.
            if (
                name_lower.endswith(".m")
                and fsize is not None
                and fsize <= MAX_SINGLE_FILE
                and matlab_bytes_total < MAX_MATLAB_BYTES
            ):
                try:
                    content = full.read_text(encoding="utf-8", errors="replace")
                    matlab_bytes_total += len(content.encode("utf-8", errors="replace"))
                    matlab_parts.append(f"\n% --- {rel} ---\n" + content)
                    if _is_top_or_first_level(rel) and _matlab_is_main_script(content, rel, readme):
                        matlab_main_scripts.append(rel)
                except Exception:
                    pass

            # Domain table headers: CSV/TSV/XLSX data-only repositories can still
            # expose interoperable chemistry or battery-performance fields.
            if (
                suffix in TABLE_EXTS
                and fsize is not None
                and fsize <= MAX_TABLE_HEADER_BYTES
                and table_header_files < MAX_TABLE_HEADER_FILES
            ):
                header = _read_table_header(full, rel)
                table_header_files += 1
                ok, matches = _domain_header_matches(header)
                if ok:
                    table_d2_headers.append(f"{rel}: {', '.join(matches)}")

            # Structured computational-materials JSON metadata. This deliberately
            # ignores NLP/text-corpus JSON such as GIVEN_TEXT/RESULT datasets.
            if (
                name_lower.endswith(".json")
                and fsize is not None
                and fsize <= MAX_JSON_EVIDENCE_BYTES
                and json_bytes_total < MAX_JSON_TEXT_BYTES
                and _json_has_pymatgen_mson(full)
            ):
                try:
                    content = full.read_text(encoding="utf-8", errors="replace")
                    json_bytes_total += len(content.encode("utf-8", errors="replace"))
                    mson_parts.append(f"\n# --- {rel} ---\n" + content[:MAX_JSON_EVIDENCE_BYTES])
                    if len(mson_files) < MAX_MSON_EVIDENCE_FILES:
                        mson_files.append(rel)
                except Exception:
                    pass

            # README
            if name_lower.startswith("readme") and not readme and fsize is not None and fsize <= MAX_SINGLE_FILE:
                try:
                    readme = full.read_text(encoding="utf-8", errors="replace")
                    if matlab_parts and not matlab_main_scripts:
                        # A README may explicitly point to a MATLAB script after
                        # the script was already scanned.
                        for part in list(matlab_parts):
                            marker = re.search(r"\n% --- (.*?) ---\n", part)
                            if marker:
                                rel_m = marker.group(1)
                                text_m = part[marker.end():]
                                if (
                                    _is_top_or_first_level(rel_m)
                                    and _matlab_is_main_script(text_m, rel_m, readme)
                                    and rel_m not in matlab_main_scripts
                                ):
                                    matlab_main_scripts.append(rel_m)
                except Exception:
                    pass

            # Dependency files — collect ALL of them
            if is_dep_file and fsize is not None and fsize <= MAX_SINGLE_FILE:
                try:
                    dep_parts.append(f"\n# --- {rel} ---\n" + full.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    pass

            # Python files — collect all (up to byte cap)
            if name_lower.endswith(".py") and fsize is not None and fsize <= MAX_SINGLE_FILE and py_bytes_total < MAX_PY_BYTES:
                try:
                    content = full.read_text(encoding="utf-8", errors="replace")
                    py_bytes_total += len(content.encode("utf-8", errors="replace"))
                    py_parts.append(f"\n# --- {rel} ---\n" + content)
                except Exception:
                    pass

    return {
        "files": files,
        "tree_data": tree_data,
        "readme": readme,
        "dep_text": "".join(dep_parts),
        "py_text": "".join(py_parts),
        "notebook_text": "".join(notebook_parts),
        "matlab_text": "".join(matlab_parts),
        "mson_text": "".join(mson_parts),
        "chem_files": chem_files,
        "mson_files": mson_files,
        "executable_notebooks": executable_notebooks,
        "matlab_main_scripts": matlab_main_scripts,
        "table_d2_headers": table_d2_headers,
    }


def resolve_repo_source(paper: dict) -> Tuple[str, str, str]:
    """Determine the best clone source for a paper.
    Returns (source_type, url, cache_name).
    source_type: 'github' | 'zenodo' | 'codeocean' | 'other' | 'none'."""
    pid = paper["paper_id"]
    if pid in SOURCE_OVERRIDES:
        return SOURCE_OVERRIDES[pid]

    gh_str = paper.get("github_urls", "").strip()
    zen_str = paper.get("zenodo_urls", "").strip()
    other_str = paper.get("other_urls", "").strip()

    # Priority 1: GitHub
    if gh_str:
        for url in gh_str.split(";"):
            url = url.strip().rstrip("/")
            url_clean = re.sub(r'/tree/.*$', '', url)
            m = re.match(r'https?://github\.com/([^/]+)/([^/]+)', url_clean)
            if m:
                owner, repo = m.group(1), m.group(2)
                repo = repo.replace(".git", "")
                cache_name = f"gh_{owner}__{repo}"
                clone_url = f"https://github.com/{owner}/{repo}.git"
                return ("github", clone_url, cache_name)

    # Priority 2: Zenodo
    if zen_str:
        for url in zen_str.split(";"):
            url = url.strip()
            if "zenodo" in url:
                cache_name = f"zen_{pid}"
                return ("zenodo", url, cache_name)

    # Priority 3: Other URLs
    if other_str:
        for url in other_str.split(";"):
            url = url.strip()
            if "codeocean.com" in url:
                return ("codeocean", url, "")
            # Generic git-cloneable URL?
            if url.endswith(".git") or "gitlab" in url or "bitbucket" in url:
                cache_name = f"other_{pid}"
                return ("other", url, cache_name)

    return ("none", "", "")


def _make_zero_result(paper: dict, note: str, github_url: str = "",
                      owner: str = "", repo: str = "") -> dict:
    """Generate an all-False result row."""
    return {
        "paper_id": paper["paper_id"],
        "sheet": paper["sheet"],
        "title": paper["top_title"][:80],
        "code_status": paper["code_status"],
        "github_url": github_url,
        "owner": owner, "repo": repo,
        "R1": False, "R2": False, "R3": False, "R4": False,
        "R5": False, "R6": False, "R7": False,
        "D1": False, "D2": False, "D3": False,
        "RRL": 0, "CRL": 0,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Indicator checkers (R2-R7, D1-D3)
# ---------------------------------------------------------------------------


def check_r2(files: list[str], readme: str) -> bool:
    """R2: Has dependency info.
    Expanded: standard dep files OR README contains 'pip install xxx'/'conda install xxx'
    with a concrete package name, OR non-standard naming (pip_environment.txt etc.)."""
    dep_files = {
        "requirements.txt", "environment.yml", "environment.yaml",
        "setup.py", "setup.cfg", "pyproject.toml",
        "Pipfile", "conda.yml", "conda.yaml",
    }
    # Standard dep files
    if any(pathlib.PurePosixPath(f).name in dep_files for f in files):
        return True
    # Non-standard dep file names (e.g. pip_environment.txt, req.txt, deps.txt)
    for f in files:
        name = pathlib.PurePosixPath(f).name.lower()
        if ("requirement" in name or "pip_env" in name or "conda_env" in name) and name.endswith(".txt"):
            return True
    # README contains concrete pip/conda install commands (not just "pip install .")
    # Must have a package name after install, e.g. "pip install torch" or "conda install rdkit"
    if re.search(r'(?:pip|pip3)\s+install\s+(?!-e\s+\.)(?!\.)[a-zA-Z][\w\-]+', readme):
        return True
    if re.search(r'conda\s+install\s+(?:-[cy]\s+\S+\s+)?[a-zA-Z][\w\-]+', readme):
        return True
    return False


def check_r3(files: list[str], readme: str) -> bool:
    """R3: Has run instructions (shell commands in README or .ipynb files)."""
    has_notebook = any(f.endswith(".ipynb") for f in files)
    has_shell_block = bool(re.search(r'```(bash|sh|shell|console|python)', readme, re.I))
    has_run_cmd = bool(re.search(r'(pip install|python |conda |pip |bash |sh )', readme))
    return has_notebook or has_shell_block or has_run_cmd


def check_r4(files: list[str]) -> bool:
    """R4: Has examples/tests.
    Expanded: also checks for .ipynb in root or first-level subdirectories."""
    test_dirs = {"examples", "tests", "test", "demo", "demos", "tutorials", "tutorial"}
    for f in files:
        parts = pathlib.PurePosixPath(f).parts
        if any(p.lower() in test_dirs for p in parts):
            return True
        name = pathlib.PurePosixPath(f).name.lower()
        if name.startswith("test_") or name.startswith("example") or name.endswith("_test.py"):
            return True
        if "demo" in name and name.endswith((".py", ".ipynb")):
            return True
        # .ipynb in root directory or first-level subdirectory
        if name.endswith(".ipynb") and len(parts) <= 2:
            return True
    return False


def check_r5(files: list[str], readme: str) -> bool:
    """R5: Has pretrained weights or download links."""
    weight_exts = {".pt", ".pth", ".ckpt", ".h5", ".pkl", ".safetensors", ".bin", ".pb"}
    for f in files:
        if pathlib.PurePosixPath(f).suffix in weight_exts:
            return True
    # Check README for download links
    if re.search(r'(pretrained|checkpoint|model.*weight|download.*model|\.pt\b|\.pth\b|\.ckpt\b)', readme, re.I):
        return True
    return False


def check_r6(files: list[str], readme: str, tree_data: Optional[list] = None) -> bool:
    """R6: Has data or data entry point.
    Tightened: data/ dir must contain actual data files (not just .gitkeep/README);
    standalone .csv/.json/.xlsx must be >10KB (uses tree_data with sizes if available)."""
    data_exts = {".csv", ".json", ".xlsx", ".parquet", ".hdf5", ".npz", ".npy"}
    config_names = {"config.json", "package.json", "package-lock.json", "tsconfig.json",
                    ".eslintrc.json", "settings.json", "manifest.json", "composer.json"}

    # Build size lookup from tree_data if available
    size_map = {}
    if tree_data:
        for item in tree_data:
            if item.get("type") == "blob" and "size" in item:
                size_map[item["path"]] = item["size"]

    for f in files:
        parts = pathlib.PurePosixPath(f).parts
        fname = pathlib.PurePosixPath(f).name
        suffix = pathlib.PurePosixPath(f).suffix

        # data/ directory with actual data files
        if "data" in [p.lower() for p in parts]:
            if fname.lower() not in {".gitkeep", ".gitignore", "readme.md", "readme", "__init__.py"}:
                # If it's a data file extension, accept
                if suffix in data_exts or suffix in {".cif", ".xyz", ".poscar", ".vasp", ".sdf"}:
                    return True
                # Other files in data/ dir are likely data too (e.g. .dat, .txt with data)
                if suffix not in {".py", ".md", ".rst", ".yml", ".yaml"}:
                    return True

        # Standalone data files outside data/ — must be >10KB if size info available
        if suffix in data_exts and fname.lower() not in config_names:
            if f in size_map:
                if size_map[f] > 10240:  # >10KB
                    return True
            else:
                # No size info — accept if not in root (likely real data, not config)
                if len(parts) > 1:
                    return True

    # Check README for data download links (Zenodo, HuggingFace, figshare)
    if re.search(r'(zenodo\.org|figshare\.com|huggingface\.co|download.*data|data.*available.*http)', readme, re.I):
        return True
    return False


def check_r7(files: list[str]) -> bool:
    """R7: Has containerization/env lock."""
    container_files = {
        "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
        ".devcontainer", "devcontainer.json",
        "conda-lock.yml", "poetry.lock", "Pipfile.lock",
    }
    for f in files:
        name = pathlib.PurePosixPath(f).name
        if name in container_files:
            return True
        if "devcontainer" in f.lower():
            return True
    return False

def check_d1(files: List[str], dep_text: str, py_text: str, readme: str,
             notebook_text: str = "", matlab_text: str = "") -> bool:
    """D1: Parameter/task configurability.

    Scores evidence that a workflow wrapper can change material system, input
    data, or task parameters without editing source code. This includes generic
    computational-chemistry ecosystems, explicit config loading plus config
    files, or stable argument parsing that exposes file/config parameters."""
    combined = (dep_text + py_text + notebook_text + matlab_text + readme).lower()
    code_text = py_text + "\n" + notebook_text + "\n" + matlab_text

    # --- Path A: ecosystem membership ---
    ecosystems = ["deepmd", "nequip", "aiida", "mace-torch", "mace_torch",
                  "schnet", "e3nn", "allegro", "dp_train", "dpdata",
                  "chgnet", "m3gnet", "mp-api", "mp_api", "matgl"]
    for eco in ecosystems:
        if eco in combined:
            return True
    # LAMMPS pair_style integration
    if re.search(r'lammps', combined) and re.search(r'(pair_style|pair_coeff|in\.lammps)', combined):
        return True

    # --- Path B: config-loading mechanism ---
    config_load_patterns = [
        r'yaml\.(?:safe_)?load',
        r'json\.load\(',
        r'(?:load|read|parse)_config',
        r'config.*=.*(?:yaml|json)\.(?:safe_)?load',
        r'OmegaConf',
        r'hydra',
        r'configparser',
    ]
    has_config_load = any(re.search(pat, code_text) for pat in config_load_patterns)
    has_config_file = any(
        ("config" in f.lower() or "cfg" in f.lower() or "params" in f.lower())
        and pathlib.PurePosixPath(f).suffix in {".yaml", ".yml", ".json", ".toml", ".cfg", ".ini"}
        for f in files
    )
    if has_config_load and has_config_file:
        return True

    # --- Path C: command-line/config arguments expose replaceable inputs ---
    if re.search(r'argparse\.ArgumentParser|parser\.add_argument', code_text):
        if re.search(r'--(?:config|input|data|dataset|structure|material|params?)\b', code_text):
            return True

    # --- Path D: Notebook/MATLAB external data entry ---
    if _code_has_external_input_loader(notebook_text) or _code_has_external_input_loader(matlab_text):
        return True

    return False


def check_d2(dep_text: str, py_text: str, readme: str,
             notebook_text: str = "", mson_text: str = "",
             chem_files: Optional[List[str]] = None,
             mson_files: Optional[List[str]] = None,
             table_d2_headers: Optional[List[str]] = None) -> bool:
    """D2: Data interoperability.

    Scores evidence for exchangeable chemistry/materials data representation:
    standard chemistry libraries in code/notebooks, explicit standard structure
    or molecule files, or structured pymatgen/ASE JSON metadata. README-only
    prose is intentionally not enough except when paired with dependency/code
    evidence."""
    combined = (dep_text + py_text + notebook_text + mson_text).lower()

    # Core chemistry I/O libraries
    libs = ['pymatgen', 'rdkit', 'openbabel', 'mdanalysis', 'matminer']
    for lib in libs:
        if lib in combined:
            return True
    # ASE: avoid false positives from words like "based", "ease", "release"
    if re.search(r'(?:import\s+ase|from\s+ase[\s.]|ase\.io|ase\.atoms|ase\.build|\base[=><])', combined):
        return True
    if chem_files:
        return True
    if mson_files:
        return True
    if table_d2_headers:
        return True

    return False


def check_d3(files: List[str], py_text: str, dep_text: str,
             executable_notebooks: Optional[List[str]] = None,
             matlab_main_scripts: Optional[List[str]] = None) -> bool:
    """D3: Interface invocability.

    Scores a stable external entry point: real CLI parser/options, console
    scripts, click/fire/typer commands, or an installable Python package. Bare
    main.py, README-only dependency mentions, and plain `if __name__` do not
    count; root/first-level executable notebooks and MATLAB main scripts do."""
    # --- Path A: CLI framework ---
    cli_patterns = [r'argparse\.ArgumentParser', r'parser\.add_argument',
                    r'click\.command', r'fire\.Fire', r'typer\.Typer',
                    r'click\.group', r'click\.option']
    for pat in cli_patterns:
        if re.search(pat, py_text):
            return True
    # console_scripts entry point
    if re.search(r'console_scripts', dep_text):
        return True

    # --- Path B: installable package ---
    install_files = {"setup.py", "setup.cfg", "pyproject.toml"}
    has_install = any(pathlib.PurePosixPath(f).name in install_files for f in files)
    has_init = any(f.endswith("__init__.py") for f in files)
    if has_install and has_init:
        return True

    # --- Path C: lightweight executable scientific workflows ---
    if executable_notebooks:
        return True
    if matlab_main_scripts:
        return True

    return False


# ---------------------------------------------------------------------------
# Scoring functions (additive — additive)
# ---------------------------------------------------------------------------
def compute_rrl(R: dict) -> int:
    """Additive RRL: R1 is prerequisite, R2+R3+R4+R6 are core (0-4),
    R5+R7 are bonus (≥1 bonus → +1, capped at 5)."""
    if not R["R1"]:
        return 0
    core = int(R["R2"]) + int(R["R3"]) + int(R["R4"]) + int(R["R6"])  # 0-4
    bonus = int(R["R5"]) + int(R["R7"])  # 0-2
    return min(5, core + (1 if bonus >= 1 else 0))


def compute_crl(D: dict) -> int:
    """CRL static: simple sum of D1+D2+D3 (0-3)."""
    return int(D["D1"]) + int(D["D2"]) + int(D["D3"])


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------
def parse_github_urls(url_str: str) -> list[tuple[str, str]]:
    """Extract (owner, repo) pairs from semicolon-separated GitHub URLs."""
    if not url_str:
        return []
    results = []
    for url in url_str.split(";"):
        url = url.strip().rstrip("/")
        # Remove /tree/... suffix
        url = re.sub(r'/tree/.*$', '', url)
        m = re.match(r'https?://github\.com/([^/]+)/([^/]+)', url)
        if m:
            owner, repo = m.group(1), m.group(2)
            repo = repo.replace(".git", "")
            results.append((owner, repo))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="RRL/CRL objective scoring (local-clone)")
    parser.add_argument("--resume", action="store_true", help="Skip already-scored repos")
    parser.add_argument("--dry-run", action="store_true", help="Show repos & source resolution")
    parser.add_argument("--clean", action="store_true", help="Delete cache after scoring")
    parser.add_argument("--timeout", type=int, default=120, help="Clone timeout in seconds")
    args = parser.parse_args()

    # Read CSV
    papers = []
    with open(CSV_IN, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cs = row["code_status"].strip().lower()
            if cs in ("yes", "partial"):
                papers.append(row)

    print(f"Found {len(papers)} papers with code=yes|partial")

    if args.dry_run:
        for p in papers:
            src_type, src_url, cache_name = resolve_repo_source(p)
            print(f"  {p['paper_id']} | {src_type:10s} | {cache_name or src_url[:60]}")
        return

    # Ensure cache dir exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing results for resume
    existing = {}
    if args.resume and CSV_OUT.exists():
        with open(CSV_OUT, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing[row["paper_id"]] = row
        print(f"Loaded {len(existing)} existing results for resume")

    # Process
    results = []

    for i, paper in enumerate(papers):
        pid = paper["paper_id"]

        if pid in FORCE_UNAVAILABLE:
            gh_url = ""
            if "github.com" in FORCE_UNAVAILABLE[pid]:
                gh_url = FORCE_UNAVAILABLE[pid].replace("Clone/download failed: ", "").replace(".git", "")
            result = _make_zero_result(paper, note=FORCE_UNAVAILABLE[pid], github_url=gh_url)
            results.append(result)
            print(f"[{i+1}/{len(papers)}] {pid} — forced unavailable → RRL=0, CRL=0")
            _save_csv(results)
            continue

        if args.resume and pid in existing:
            results.append(existing[pid])
            print(f"[{i+1}/{len(papers)}] {pid} — skipped (resume)")
            continue

        src_type, src_url, cache_name = resolve_repo_source(paper)

        # --- No cloneable source ---
        if src_type == "none":
            result = _make_zero_result(paper, note="No cloneable URL")
            results.append(result)
            print(f"[{i+1}/{len(papers)}] {pid} — no URL → RRL=0, CRL=0")
            _save_csv(results)
            continue

        if src_type == "codeocean":
            result = _make_zero_result(paper, note=f"Code Ocean (manual): {src_url[:80]}")
            results.append(result)
            print(f"[{i+1}/{len(papers)}] {pid} — Code Ocean (cannot auto-clone) → RRL=0, CRL=0")
            _save_csv(results)
            continue

        # --- Clone / download ---
        dest_dir = CACHE_DIR / cache_name
        print(f"[{i+1}/{len(papers)}] {pid} — {src_type}: {cache_name}")

        if src_type == "zenodo":
            ok = download_zenodo(src_url, dest_dir)
        else:
            ok = clone_repo(src_url, dest_dir, timeout=args.timeout)

        if not ok:
            # R1 = False (not accessible)
            gh_url = src_url.replace(".git", "") if src_type == "github" else ""
            result = _make_zero_result(paper, note=f"Clone/download failed: {src_url[:80]}",
                                       github_url=gh_url)
            results.append(result)
            print(f"  → R1=False → RRL=0")
            _save_csv(results)
            continue

        # --- Scan local repo ---
        scan = scan_local_repo(dest_dir)
        files = scan["files"]
        tree_data = scan["tree_data"]
        readme = scan["readme"]
        dep_text = scan["dep_text"]
        py_text = scan["py_text"]
        notebook_text = scan["notebook_text"]
        matlab_text = scan["matlab_text"]
        mson_text = scan["mson_text"]
        chem_files = scan["chem_files"]
        mson_files = scan["mson_files"]
        executable_notebooks = scan["executable_notebooks"]
        matlab_main_scripts = scan["matlab_main_scripts"]
        table_d2_headers = scan["table_d2_headers"]

        # Derive owner/repo for output
        owner, repo_name = "", ""
        if src_type == "github":
            m = re.match(r'https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?$', src_url)
            if m:
                owner, repo_name = m.group(1), m.group(2)
        gh_url = f"https://github.com/{owner}/{repo_name}" if owner else src_url

        # --- All RRL indicators ---
        R = {
            "R1": True,
            "R2": check_r2(files, readme),
            "R3": check_r3(files, readme),
            "R4": check_r4(files),
            "R5": check_r5(files, readme),
            "R6": check_r6(files, readme, tree_data),
            "R7": check_r7(files),
        }

        # --- All CRL indicators (static: D1/D2/D3) ---
        d1 = check_d1(files, dep_text, py_text, readme, notebook_text, matlab_text)
        d2 = check_d2(dep_text, py_text, readme, notebook_text, mson_text,
                      chem_files, mson_files, table_d2_headers)
        d3 = check_d3(files, py_text, dep_text, executable_notebooks, matlab_main_scripts)
        D = {"D1": d1, "D2": d2, "D3": d3}

        rrl = compute_rrl(R)
        crl = compute_crl(D)

        result = {
            "paper_id": pid,
            "sheet": paper["sheet"],
            "title": paper["top_title"][:80],
            "code_status": paper["code_status"],
            "github_url": gh_url,
            "owner": owner, "repo": repo_name,
            **{k: v for k, v in R.items()},
            **{k: v for k, v in D.items()},
            "RRL": rrl, "CRL": crl,
            "note": f"src={src_type}"
        }
        results.append(result)
        print(f"  → RRL={rrl}, CRL={crl}  R={[int(v) for v in R.values()]}  D=[{d1},{d2},{d3}]")

        # Save incrementally (for resume)
        _save_csv(results)

    # Apply manual overrides
    if OVERRIDES_CSV.exists():
        with open(OVERRIDES_CSV, encoding="utf-8") as f:
            overrides = {row["paper_id"]: row for row in csv.DictReader(f)}
        applied = 0
        for r in results:
            pid = r["paper_id"]
            if pid in overrides:
                for k, v in overrides[pid].items():
                    if k == "rationale":
                        continue  # rationale is documentation only
                    r[k] = v
                applied += 1
        print(f"\nApplied {applied} manual overrides from {OVERRIDES_CSV.name}")

    # Final save
    _save_csv(results)
    print(f"\nDone. {len(results)} repos scored.")
    print(f"   Output: {CSV_OUT}")

    # Summary stats
    rrls = [int(r["RRL"]) for r in results]
    crls = [int(r["CRL"]) for r in results]
    print(f"\nRRL distribution:")
    for v in range(6):
        n = rrls.count(v)
        print(f"   RRL-{v}: {n} ({100*n/len(rrls):.1f}%)")
    print(f"\nCRL distribution:")
    for v in range(4):
        n = crls.count(v)
        print(f"   CRL-{v}: {n} ({100*n/len(crls):.1f}%)")

    import statistics
    print(f"\n   RRL median: {statistics.median(rrls)}")
    print(f"   CRL median: {statistics.median(crls)}")

    # Clean cache if requested
    if args.clean and CACHE_DIR.exists():
        print(f"\nCleaning cache: {CACHE_DIR}")
        shutil.rmtree(CACHE_DIR, ignore_errors=True)


def _save_csv(results: list[dict]):
    """Save results to CSV."""
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
