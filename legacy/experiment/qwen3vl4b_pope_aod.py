"""Select and evaluate AOD on Qwen3-VL-4B using the fixed POPE split."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

import run as core
from qwen_layerstudy_v28 import direction, steer


LAYERS = list(range(16, 29))
ALPHAS = [-1.0, -0.5, 0.5, 1.0]
CS = [0.001, 0.01, 0.1, 1.0]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_linear(train_x, train_y, eval_x, c):
    center = train_x.mean(0)
    scale = (train_x - center).square().mean().sqrt().clamp_min(1e-6)
    model = LogisticRegression(C=c, class_weight="balanced", solver="liblinear",
                               max_iter=4000, random_state=42)
    model.fit(((train_x - center) / scale).numpy(), train_y)
    score = model.decision_function(((eval_x - center) / scale).numpy())
    payload = {"coef": torch.tensor(model.coef_[0]), "intercept": float(model.intercept_[0]),
               "center": center, "scale": scale, "C": c}
    return score, payload


def apply_linear(payload, values):
    x = (values - payload["center"]) / payload["scale"]
    return (x @ payload["coef"].to(x.dtype) + payload["intercept"]).numpy()


def metrics(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {"n": len(y), "accuracy": 100 * (tp + tn) / len(y),
            "precision": 100 * tp / max(1, tp + fp), "recall": 100 * tp / max(1, tp + fn),
            "f1": 200 * tp / max(1, 2 * tp + fp + fn), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "false_positive_rate": 100 * fp / max(1, fp + tn),
            "false_discovery_rate": 100 * fp / max(1, tp + fp)}


def threshold_candidates(scores):
    values = np.unique(scores)
    return np.r_[values[0] - 1, (values[:-1] + values[1:]) / 2, values[-1] + 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("search", "test"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")

    feature_path = core.ROOT / "results/qwen3vl4b_pope/features.pt"
    cache = torch.load(feature_path, map_location="cpu", weights_only=False)
    rows = cache["records"]
    assert len(rows) == 3000 and sorted(cache["hidden"]) == LAYERS
    ids = {s: [i for i, r in enumerate(rows) if r["split"] == s]
           for s in ("train", "val", "test")}
    assert {k: len(v) for k, v in ids.items()} == {"train": 1800, "val": 600, "test": 600}
    assert not {rows[i]["group"] for i in ids["train"]} & {rows[i]["group"] for i in ids["test"]}
    source_sha = sha(feature_path)

    if args.stage == "search":
        assert not args.out.exists()
        args.out.mkdir(parents=True)
        train_y = np.array([rows[i]["gt"] for i in ids["train"]])
        val_y = np.array([rows[i]["gt"] for i in ids["val"]])
        train_base = np.array([rows[i]["pred"] for i in ids["train"]])
        val_base = np.array([rows[i]["pred"] for i in ids["val"]])
        train_correct = torch.tensor(train_base == train_y, dtype=torch.float32, device="cuda")
        val_correct = torch.tensor(val_base == val_y, dtype=torch.float32, device="cuda")
        val_margin = np.array([rows[i]["margin"] for i in ids["val"]])
        candidates = []
        direction_files = []
        for layer in LAYERS:
            values = torch.stack(cache["hidden"][layer]).float()
            vector, checkpoint = direction(values[ids["train"]].cuda(), train_correct,
                                           values[ids["val"]].cuda(), val_correct, layer,
                                           [rows[i]["id"] for i in ids["train"]])
            vector = vector.cpu()
            direction_path = args.out / f"direction_L{layer}.pt"
            torch.save(checkpoint, direction_path)
            direction_files.append({"path": str(direction_path), "sha256": sha(direction_path)})
            for alpha, c in itertools.product(ALPHAS, CS):
                score, _ = fit_linear(steer(values[ids["train"]], vector, alpha), train_y,
                                      steer(values[ids["val"]], vector, alpha), c)
                for blend in (0.75, 1.0):
                    combined = blend * score + (1 - blend) * val_margin / 4
                    for threshold in threshold_candidates(combined):
                        pred = (combined >= threshold).astype(int)
                        metric = metrics(val_y, pred)
                        # The acceptance constraints are primary; within them maximize F1,
                        # then accuracy and retain a non-zero orthogonal intervention.
                        passed = metric["f1"] >= 90 and metric["false_positive_rate"] <= 10
                        candidates.append({"layer": layer, "alpha": alpha, "C": c,
                                           "blend": blend, "threshold": float(threshold),
                                           "validation": metric, "passes": passed})
            print("SEARCH", layer, flush=True)
        candidates.sort(key=lambda c: (c["passes"], c["validation"]["f1"],
                                       c["validation"]["accuracy"], -c["validation"]["false_positive_rate"],
                                       -abs(c["alpha"]), -c["layer"]), reverse=True)
        selected = candidates[0]
        core.atomic_json(args.out / "selection.json", selected)
        core.atomic_json(args.out / "validation_search.json", candidates[:500])
        core.atomic_json(args.out / "protocol.json", {
            "model": "Qwen3-VL-4B-Instruct", "method": "AOD", "feature_sha256": source_sha,
            "layers": LAYERS, "alphas": ALPHAS, "C": CS,
            "split": "fixed group-disjoint 1800 train / 600 validation / 600 test",
            "selection": "validation only; maximize F1 subject to F1>=90 and FPR<=10",
            "test_used_for_selection": False, "direction_files": direction_files,
        })
        print("SELECTED", json.dumps(selected), flush=True)
        return

    protocol = json.loads((args.out / "protocol.json").read_text())
    selected = json.loads((args.out / "selection.json").read_text())
    assert protocol["feature_sha256"] == source_sha and protocol["test_used_for_selection"] is False
    assert not (args.out / "results.json").exists()
    train_dev = ids["train"] + ids["val"]
    train_y = np.array([rows[i]["gt"] for i in train_dev])
    train_base = np.array([rows[i]["pred"] for i in train_dev])
    train_correct = torch.tensor(train_base == train_y, dtype=torch.float32, device="cuda")
    values = torch.stack(cache["hidden"][selected["layer"]]).float()
    vector, checkpoint = direction(values[train_dev].cuda(), train_correct,
                                   values[ids["test"][:0]].cuda(), train_correct[:0], selected["layer"],
                                   [rows[i]["id"] for i in train_dev])
    vector = vector.cpu()
    direction_path = args.out / "frozen_direction.pt"
    torch.save(checkpoint, direction_path)
    _, payload = fit_linear(steer(values[train_dev], vector, selected["alpha"]), train_y,
                            steer(values[ids["test"]], vector, selected["alpha"]), selected["C"])
    score = apply_linear(payload, steer(values[ids["test"]], vector, selected["alpha"]))
    margin = np.array([rows[i]["margin"] for i in ids["test"]])
    combined = selected["blend"] * score + (1 - selected["blend"]) * margin / 4
    pred = (combined >= selected["threshold"]).astype(int)
    model_path = args.out / "frozen_answer_head.pt"
    payload.update({"direction": vector, "layer": selected["layer"], "alpha": selected["alpha"],
                    "train_ids": [rows[i]["id"] for i in train_dev]})
    torch.save(payload, model_path)
    test_y = np.array([rows[i]["gt"] for i in ids["test"]])
    base_pred = np.array([rows[i]["pred"] for i in ids["test"]])
    base_metrics = metrics(test_y, base_pred)
    aod_metrics = metrics(test_y, pred)
    records = [dict(rows[i], base_pred=int(b), pred=int(p), classifier_logit=float(s),
                    combined_score=float(z))
               for i, b, p, s, z in zip(ids["test"], base_pred, pred, score, combined)]
    result = {"model": "Qwen3-VL-4B-Instruct", "dataset": "POPE", "method": "AOD",
              "counts": {k: len(v) for k, v in ids.items()}, "configuration": selected,
              "base": base_metrics, "aod": aod_metrics,
              "f1_gain_pp": aod_metrics["f1"] - base_metrics["f1"],
              "criterion_met": aod_metrics["f1"] >= 90 and aod_metrics["false_positive_rate"] <= 10,
              "test_used_for_selection": False,
              "artifacts": {"direction": {"path": str(direction_path), "sha256": sha(direction_path)},
                            "head": {"path": str(model_path), "sha256": sha(model_path)}}}
    core.atomic_json(args.out / "test_predictions.json", records)
    core.atomic_json(args.out / "results.json", result)
    print("POPE TEST RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
