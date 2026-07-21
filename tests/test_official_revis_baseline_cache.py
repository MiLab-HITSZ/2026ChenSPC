import json
import tempfile
from pathlib import Path

from evaluate_cdh_bench import _load_official_revis_baseline_cache


def test_load_official_revis_baseline_cache():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "results.jsonl"
        row = {
            "status": "ok",
            "pair_id": "Pair 1",
            "task": "qa",
            "side": "counterfactual",
            "revis": {"baseline_prediction": "yes", "steered_prediction": "no"},
            "raw": {"baseline_raw": {"steering": False}},
        }
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")

        cache = _load_official_revis_baseline_cache(str(path))

        prediction, raw = cache[("Pair 1", "qa", "counterfactual")]
        assert prediction == "yes"
        assert raw == {"steering": False}
