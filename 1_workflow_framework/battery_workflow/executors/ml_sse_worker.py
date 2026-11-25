"""Standalone ML-SSE inference worker.

Run under the sklearn that can unpickle the released RandomForest trees
(sklearn 1.0.2). The parent process
sends a JSON object on stdin and reads a JSON array of log10(sigma) values
from stdout. This file must not import ``battery_workflow`` -- the fallback
interpreter does not have this repository on its default path.
"""

from __future__ import annotations

import json
import os
import sys
import warnings


def main() -> int:
    payload = json.load(sys.stdin)
    repo = payload["repo"]
    formulas = list(payload["formulas"])
    sys.path.insert(0, repo)
    os.chdir(repo)
    warnings.filterwarnings("ignore")
    import joblib
    import pandas as pd

    model = joblib.load(os.path.join(repo, "model.joblib"))
    dummy = pd.DataFrame({"formula": formulas, "target": [0.0] * len(formulas)})
    predictions = [float(x) for x in model.predict(dummy)]
    json.dump(predictions, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
