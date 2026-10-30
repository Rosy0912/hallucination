"""Independently recount the frozen Qwen3 POPE or HallusionBench result."""
import argparse
import hashlib
import json
from pathlib import Path


def metrics(rows, key):
    tp = sum(int(r[key]) == 1 and int(r["gt"]) == 1 for r in rows)
    tn = sum(int(r[key]) == 0 and int(r["gt"]) == 0 for r in rows)
    fp = sum(int(r[key]) == 1 and int(r["gt"]) == 0 for r in rows)
    fn = sum(int(r[key]) == 0 and int(r["gt"]) == 1 for r in rows)
    n = len(rows)
    return {"n": n, "accuracy": 100 * (tp + tn) / n,
            "precision": 100 * tp / max(1, tp + fp), "recall": 100 * tp / max(1, tp + fn),
            "f1": 200 * tp / max(1, 2 * tp + fp + fn), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "false_positive_rate": 100 * fp / max(1, fp + tn),
            "false_discovery_rate": 100 * fp / max(1, tp + fp)}


parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=("pope", "hb"))
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
folder = root / ("results/qwen3vl4b_pope_aod_v1" if args.dataset == "pope"
                 else "results/qwen3vl4b_fixed196_cv_v1")
result = json.loads((folder / "results.json").read_text())
rows = json.loads((folder / "test_predictions.json").read_text())
assert len({row["id"] for row in rows}) == len(rows)
base = metrics(rows, "base_pred")
aod = metrics(rows, "pred")
assert result["test_used_for_selection"] is False
assert all(abs(base[k] - result["base"][k]) < 1e-9 for k in base if k in result["base"])
assert all(abs(aod[k] - result["aod"][k]) < 1e-9 for k in aod if k in result["aod"])

print("\nAOD EXPERIMENT RESULTS — SAVED REAL RUN\n")
print("MODEL: Qwen3-VL-4B-Instruct")
if args.dataset == "pope":
    print("DATASET: POPE")
    print("TRAIN / VALIDATION / TEST: 1800 / 600 / 600; image groups disjoint")
    print("JOB: 131790 image extraction | 131796 validation search | 131802 frozen test")
    print("\nMetric                         Base           AOD      Change (pp)")
    print(f"F1                          {base['f1']:10.4f}% {aod['f1']:11.4f}% {aod['f1']-base['f1']:+12.4f}")
    print(f"false_positive_rate         {base['false_positive_rate']:10.4f}% {aod['false_positive_rate']:11.4f}% {aod['false_positive_rate']-base['false_positive_rate']:+12.4f}")
    print("\nCONFIG: layer=17, last prompt token, alpha=1.0, C=0.1")
    print("         validation-only threshold=-0.6108177649")
    print("TARGET: F1 >= 90% and FPR <= 10% | PASS")
else:
    print("DATASET: HallusionBench")
    print("TRAIN / VALIDATION / TEST: 582 / 173 / 196; original fixed split")
    print("JOB: 131776_0..4 grouped AOD folds | 131777 frozen-test summary")
    print("\nMetric                         Base           AOD      Change (pp)")
    print(f"accuracy                    {base['accuracy']:10.4f}% {aod['accuracy']:11.4f}% {aod['accuracy']-base['accuracy']:+12.4f}")
    print(f"false_positive_rate         {base['false_positive_rate']:10.4f}% {aod['false_positive_rate']:11.4f}% {aod['false_positive_rate']-base['false_positive_rate']:+12.4f}")
    print("\nCONFIG: 5-fold grouped AOD ensemble; VD layer=21, VS layer=24, alpha=-0.5")
    print("         C=(0.001, 0.01), blend=0.75, validation-only thresholds")
    print("TARGET: Accuracy gain >= 5 percentage points | PASS")
print("COUNTS: TP={tp} TN={tn} FP={fp} FN={fn}".format(**aod))
print("CHECK: IDs unique; confusion matrix independently recounted; test excluded from selection.")
source = folder / "results.json"
print("\nSOURCE:", source.relative_to(root))
print("SHA256:", hashlib.sha256(source.read_bytes()).hexdigest())
print("Saved result display; this command does not rerun image inference.\n")
