"""
LiSSE-MP: nine AIMD-confirmed superionic conductors (audit paper_id 10664).

The virtual-experiment source for the Case A end-to-end chain. The release
ships serialized pymatgen ``DiffusionAnalyzer`` objects for nine structures at
four temperatures each, and its README labels all nine "superionic SSEs
confirmed by AIMD simulations". The feedback agent serves those published
trajectories: an exact-composition match is a published virtual experiment for
that survivor, a same-element-system match is family-level context, and
anything else is an explicit absence.

Nothing here is measured, and no trajectory is produced by this framework.
Where a conductivity number is reported downstream, it comes from
deserializing the authors' own analysis objects (``executors/lisse_mp_aimd``),
which recomputes their MSD statistics from their stored displacements -- the
same operation their notebook performs on load.

Stdlib-only, matching the other dataset modules.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from math import gcd
from pathlib import Path

from ._source import PublishedSource

SOURCE = PublishedSource(
    key="lisse_mp",
    tool="LiSSE-MP",
    paper_id="10664",
    citation="[@kang2023discovery]",
    doi="10.1021/acs.jpcc.3c02908",
    repository="https://github.com/kminmin/LiSSE-MP",
    commit="a223cc80ea98e40008c0469e409bffd4d7c2147d",
    licence="none declared (do not redistribute)",
    redistributable=False,
    env_var="BATTAGENT_LISSE_MP_DIR",
    cache_dirname="gh_kminmin__LiSSE-MP",
    files={
        "README.md":
            "473b5a5a15226eb2f3e648a89a4456b6b925c26991b1c2af349343391024193a",
        "1-mp-532413/800_nteps_100.json":
            "634935dae47ea21d95472d6c70983226c703e6b21e4c9cdcff0ce62dd3a71210",
        "1-mp-532413/1100_nteps_100.json":
            "7286f757e8e01565eb98ef33129cbe3ff422ab7d7d306c508d3edbdb9f6b2911",
        "1-mp-532413/1400_nteps_100.json":
            "b5687d5e5200638b35d7d2bd314f140a717b8da4180837bba733008fb7f93a70",
        "1-mp-532413/1700_nteps_100.json":
            "f8e1494901d1bcdd05d35182ce6151b14b57ef21a45a665d2210088d420601ad",
        "2-mp-676361/800_nteps_100.json":
            "f6fdc1adaed5d90d4a2bf122847ec2b54b0239e19142a68a7439c8daf26467ae",
        "2-mp-676361/1100_nteps_100.json":
            "c4b4c4e156c0936571f6fdfa7f92cb3074c8978a765fdc4058d024a3ce8d0639",
        "2-mp-676361/1400_nteps_100.json":
            "68b2474b7c4dc0d5b874d42795e333faaec21f9d658359bd981ca69759d1829c",
        "2-mp-676361/1700_nteps_100.json":
            "10748934846d128815a83f56a947ea29d6a236633ece5f3c9c247dd0704aed3b",
        "3-mp-29410/800_nteps_100.json":
            "238b997d8a9ca2014c2f3e82cd6185148cdf07deab972b482d8329556514a3b2",
        "3-mp-29410/1100_nteps_100.json":
            "fd0a2e48957de664fe8de662cd743eea4a4d0f7eeced4c235c04c09ada8a5fd1",
        "3-mp-29410/1400_nteps_100.json":
            "c2a818c21b9c8fbcc8d7dbc86742a6d3b9abb29ea297e4432d51bec798ec0256",
        "3-mp-29410/1700_nteps_100.json":
            "7888f27acad1f84f30285959a22776ca8ec448c661c5cffb66c5fa7370037844",
        "4-mp-1106183/800_nteps_100.json":
            "aa1d90b38e38e10e439db8aa1a186081ff06789381b5408776ec1de339745458",
        "4-mp-1106183/1100_nteps_100.json":
            "084b87d811d41c4daf119115fadaf0d8017fc2e5a50570564fcd7120ff8c36ec",
        "4-mp-1106183/1400_nteps_100.json":
            "92d3ebfd92a09968bd049e1c5e608956006e946e3f4204ce8d6028349d7a6ed1",
        "4-mp-1106183/1700_nteps_100.json":
            "ab7d44900eb98f9d00ef502a8562378f9868c4c32e6dbb2d8b5d3da125904c7e",
        "5_mp-1222492/800_nteps_100.json":
            "8466d114504b4202e029854dfac5e66a84fe5b2755db235d078561696d188fbc",
        "5_mp-1222492/1100_nteps_100.json":
            "7ed7c7f2ba15f6168c586728d4687e16b75e95d5183008622205d7abfcbb3d59",
        "5_mp-1222492/1400_nteps_100.json":
            "fb180a5f73e482946b091ba423cabfc3e2a58054db160b32fb859cf96fa88bc3",
        "5_mp-1222492/1700_nteps_100.json":
            "8300562a1bf9bbfc08171fb5fb2fca28fe43c35789c0d80a412797f61ec9a076",
        "6-mp-19896/800_nteps_100.json":
            "d42f10746a8f2ba84bbf09d1bf1061204638067bf5afd85bd6e5fac4a49afcd3",
        "6-mp-19896/1100_nteps_100.json":
            "4bf19598b95623b638b34f862e9e37314f2b7293d17edf51a7fd34408739707b",
        "6-mp-19896/1400_nteps_100.json":
            "1c77d12e34a81eb46d185e2f17cfa403afb857a8219cce0ecce32ad6f271992b",
        "6-mp-19896/1700_nteps_100.json":
            "717ce3fd2ce6d22a9181988baa0e765abc707a447bbf28ad0daa00b05ef2ddc9",
        "7-mp-1222404/800_nteps_100.json":
            "77ccc6ab3a52abb23fdf23110f4a854008883435a6d7ddb09cfbefe4afc4e726",
        "7-mp-1222404/1100_nteps_100.json":
            "5bdf070aff394b38fbe791a7808361aba4d1ad669211da70257236d629e96990",
        "7-mp-1222404/1400_nteps_100.json":
            "a793e481b94bf092e959e790399f91c4cb7714e88b1532b5ca53584af686468c",
        "7-mp-1222404/1700_nteps_100.json":
            "9762a48cd04a4e96ed86400231e0bd6720775bb838a7f1ff055fee1a6f68eebf",
        "8-mp-1195055/800_nteps_100.json":
            "4a7dff096da6c4087b4145eeaadf70b285183361f918c13ccc34212336b17ac8",
        "8-mp-1195055/1100_nteps_100.json":
            "aaee3292b8da8d383a626508f788638a82684d22f4187ee014d66816858484a2",
        "8-mp-1195055/1400_nteps_100.json":
            "738da7ed02c21a527549092e010dcc15d620a35f3ea21e0311934dcf747bebda",
        "8-mp-1195055/1700_nteps_100.json":
            "8e5352b896821c9de894bb49059416467dc4a186c16656013d4a08d2411a629a",
        "9-mp-557892/800_nteps_100.json":
            "711dcdc39d9dfdb2f130d898d0268cd270a5023037ea59bc1cd2b5a835fa0d09",
        "9-mp-557892/1100_nteps_100.json":
            "5214b5005e1777bbf744e024dae1455522b62d6c3a69289a511f889a66b2c446",
        "9-mp-557892/1400_nteps_100.json":
            "9ccce929688af5ee5859b5f56a324a2304bfba57fdcb4923fc56e81de0c25c8e",
        "9-mp-557892/1700_nteps_100.json":
            "c300caf345003cff4687b19f390108a0e02f282743f4d9f358e0bea12dafed76",
    },
)

#: Temperatures at which the release ran AIMD, kelvin. Encoded in the filenames.
TEMPERATURES_K = (800, 1100, 1400, 1700)

_DIRNAME = re.compile(r"^\d+[-_](mp-\d+)$")
_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")

#: All IUPAC element symbols. Matching against a stoichiometry key must reject
#: tokens that merely look like elements.
ELEMENT_SYMBOLS = frozenset(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I "
    "Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt "
    "Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr "
    "Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split()
)


def parse_formula(formula: str) -> dict[str, int]:
    """'Li2B1S2' -> {'Li': 2, 'B': 1, 'S': 2}. Repeated symbols accumulate.

    Strict by design: the whole string must parse, counts must be integers, and
    every symbol must be a real element. The previous permissive regex silently
    truncated decimals ('Li0.5Al0.5S1' became {'S': 1}) and accepted trailing
    garbage, which could have produced false stoichiometric matches. Feedback
    decisions must fail loudly on inputs they cannot represent rather than
    match on a misparse.
    """
    text = (formula or "").strip()
    if not text:
        raise ValueError("empty formula")
    counts: dict[str, int] = {}
    position = 0
    for match in _FORMULA_TOKEN.finditer(text):
        if match.start() != position:
            raise ValueError(
                f"cannot parse formula {formula!r}: unexpected {text[position:match.start()]!r}"
            )
        symbol, digits = match.group(1), match.group(2)
        if symbol not in ELEMENT_SYMBOLS:
            raise ValueError(f"unknown element {symbol!r} in formula {formula!r}")
        counts[symbol] = counts.get(symbol, 0) + (int(digits) if digits else 1)
        position = match.end()
    if position != len(text):
        raise ValueError(f"cannot parse formula {formula!r}: trailing {text[position:]!r}")
    return counts


def reduced_ratio(counts: dict[str, int]) -> tuple[tuple[str, int], ...]:
    """Order-free stoichiometry key: {'Li': 32, 'Al': 32, 'S': 64} -> ((Al,1),(Li,1),(S,2))."""
    positive = {el: n for el, n in counts.items() if n > 0}
    if not positive:
        return ()
    divisor = 0
    for n in positive.values():
        divisor = gcd(divisor, n)
    return tuple(sorted((el, n // divisor) for el, n in positive.items()))


@dataclass(frozen=True)
class AimdStructure:
    """One AIMD-confirmed structure of the release, composition read from its file."""

    mp_id: str
    dirname: str
    element_counts: dict[str, int]

    @property
    def element_set(self) -> frozenset[str]:
        return frozenset(self.element_counts)

    @property
    def reduced(self) -> tuple[tuple[str, int], ...]:
        return reduced_ratio(self.element_counts)

    @property
    def formula(self) -> str:
        return "".join(f"{el}{n}" for el, n in self.reduced)

    def trajectory_files(self) -> list[str]:
        return [f"{self.dirname}/{t}_nteps_100.json" for t in TEMPERATURES_K]


@dataclass(frozen=True)
class LisseMpIndex:
    root: Path
    release_claim: str
    structures: tuple[AimdStructure, ...]

    def exact_match(self, formula: str) -> AimdStructure | None:
        """The published structure with the same reduced stoichiometry, if any."""
        key = reduced_ratio(parse_formula(formula))
        for structure in self.structures:
            if structure.reduced == key:
                return structure
        return None

    def same_system(self, formula: str) -> list[AimdStructure]:
        """Published structures sharing the element set but not the stoichiometry."""
        counts = parse_formula(formula)
        elements = frozenset(el for el, n in counts.items() if n > 0)
        key = reduced_ratio(counts)
        return [
            s for s in self.structures
            if s.element_set == elements and s.reduced != key
        ]

    def trajectory_paths(self, structure: AimdStructure) -> list[Path]:
        return [self.root / relative for relative in structure.trajectory_files()]


def read_trajectory_dict(path: Path) -> dict:
    """Decode one released trajectory file into the analyzer's dict form.

    The release double-encodes each file (a JSON string holding JSON). This is
    the single decoder every consumer must use -- the AIMD executor and its
    worker import it rather than re-implementing the unwrap.
    """
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: decoded trajectory is {type(payload).__name__}, not dict")
    return payload


def _element_counts(trajectory: dict) -> dict[str, int]:
    """Integer stoichiometry from the released sites, occupancy included.

    Site occupancies are accumulated instead of assumed to be one; a structure
    with partial or mixed occupancy has no integer stoichiometry and must not
    silently enter exact/family matching. The nine pinned
    structures are all single-species, occupancy-1 sites, so this raises only
    if the pinned bytes ever change.
    """
    accumulated: dict[str, float] = {}
    for site in trajectory["structure"]["sites"]:
        for species in site["species"]:
            element = species["element"]
            occupancy = float(species.get("occu", 1))
            accumulated[element] = accumulated.get(element, 0.0) + occupancy
    counts: dict[str, int] = {}
    for element, total in accumulated.items():
        rounded = round(total)
        if abs(total - rounded) > 1e-6:
            raise ValueError(
                f"non-integer total occupancy for {element} ({total}); refusing "
                "stoichiometric matching on a disordered structure"
            )
        counts[element] = rounded
    return counts


def load(root: str | Path | None = None, verify: bool = True) -> LisseMpIndex:
    """Verify the pinned files and index the nine structures by composition.

    Compositions are read from the release's own trajectory files rather than
    hardcoded, so the index cannot drift from the bytes it is pinned to.
    """
    if verify:
        SOURCE.verify(root)
    base = SOURCE.root(root)

    claim = ""
    readme = (base / "README.md").read_text(encoding="utf-8")
    for line in readme.splitlines():
        if "superionic" in line.lower():
            claim = line.strip()
            break

    structures: list[AimdStructure] = []
    seen_dirs: list[str] = []
    for relative in SOURCE.files:
        directory = relative.split("/", 1)[0]
        if directory not in seen_dirs and _DIRNAME.match(directory):
            seen_dirs.append(directory)
    for directory in seen_dirs:
        mp_id = _DIRNAME.match(directory).group(1)
        trajectory = read_trajectory_dict(base / f"{directory}/{TEMPERATURES_K[0]}_nteps_100.json")
        structures.append(AimdStructure(
            mp_id=mp_id,
            dirname=directory,
            element_counts=_element_counts(trajectory),
        ))
    if len(structures) != 9:
        raise RuntimeError(f"expected 9 AIMD structures in the release, found {len(structures)}")

    return LisseMpIndex(root=base, release_claim=claim, structures=tuple(structures))
