"""Runnable example workflows for the battery coordination framework.

One is a deterministic trace replay against published data, two run released
models and the authors' corpus on top of a replay, and ``run_case2`` executes
the Case 2 electrolyte sharing and ablation experiments:

    ml_htvs_screening           material-level screening funnel  [@kim2023accelerating]
    ml_htvs_continuation        execute ML-SSE on that shortlist [@kurniawan2025unveiling]
    ml_htvs_case_a_full         the whole Case A chain: replay -> execute ->
                                retrieve -> suggest, five roles on one coordinator
    run_case2                   Case 2 OEDB record-sharing and ablation campaigns
"""

import sys


def force_utf8_stdout() -> None:
    """Let the traces print on a legacy Windows code page.

    Only touches the real console. Under pytest capture or ``redirect_stdout``
    the stream belongs to the caller, and replacing it there breaks their
    teardown.
    """
    stream = sys.stdout
    if stream is not sys.__stdout__ or stream is None:
        return
    encoding = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
    if encoding == "utf8":
        return
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")
