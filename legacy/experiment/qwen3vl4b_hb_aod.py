"""Grouped development selection and fixed-test AOD for Qwen3-VL-4B."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

import run as core
from qwen_layerstudy_v28 import binary, direction, steer


LAYERS = list(range(16, 25))
ALPHAS = [-1.0, -0.5, 0.0, 0.5, 1.0]
CS = [0.01, 0.1, 1.0, 10.0]
SEED = 20260929


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_linear(train_x, train_y, valid_x, c):
    center = train_x.mean(0)
    scale = (train_x - center).square().mean().sqrt().clamp_min(1e-6)
    x = ((train_x - center) / scale).numpy()
    z = ((valid_x - center) / scale).numpy()
    model = LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=3000,
                               random_state=42)
    model.fit(x, train_y)
    return model.decision_function(z), {
        "coef": torch.tensor(model.coef_[0]), "intercept": float(model.intercept_[0]),
        "center": center, "scale": scale, "C": c,
    }


def apply_linear(payload, values):
    x = (values - payload["center"]) / payload["scale"]
    coefficient = payload["coef"].to(dtype=x.dtype)
    return (x @ coefficient + payload["intercept"]).numpy()


def best_threshold(score, labels):
    values = np.unique(score)
    candidates = np.r_[values[0] - 1, (values[:-1] + values[1:]) / 2, values[-1] + 1]
    return max(candidates, key=lambda t: (np.mean((score >= t) == labels), -abs(float(t))))


def predict(score, margin, blend, threshold):
    return blend * score + (1 - blend) * margin / 4 >= threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("search", "test"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")

    feature_path = core.ROOT / "results/qwen3vl4b_hb/features.pt"
    cache = torch.load(feature_path, map_location="cpu", weights_only=False)
    rows = cache["records"]
    assert len(rows) == 951 and sorted(cache["hidden"]) == LAYERS
    selected_test = json.loads((core.ROOT / "results/qwen_resplit_v62/fold1/test_predictions.json").read_text())
    test_ids = {r["id"] for r in selected_test}
    dev = [i for i, r in enumerate(rows) if r["id"] not in test_ids]
    test = [i for i, r in enumerate(rows) if r["id"] in test_ids]
    assert len(dev) == 769 and len(test) == 182
    assert not {rows[i]["group"] for i in dev} & {rows[i]["group"] for i in test}
    y = np.array([rows[i]["gt"] for i in dev])
    base = np.array([rows[i]["pred"] for i in dev])
    margin = np.array([rows[i]["margin"] for i in dev])
    groups = np.array([rows[i]["group"] for i in dev])
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(dev)), y, groups))
    source_sha = sha(feature_path)

    if args.stage == "search":
        assert not args.out.exists()
        args.out.mkdir(parents=True)
        oof = {f"{layer}_{alpha}_{c}": np.zeros(len(dev))
               for layer, alpha, c in itertools.product(LAYERS, ALPHAS, CS)}
        direction_files = {}
        for fold, (train_ids, valid_ids) in enumerate(folds):
            train_y = y[train_ids]
            train_correct = torch.tensor(base[train_ids] == train_y, dtype=torch.float32, device="cuda")
            valid_correct = torch.tensor(base[valid_ids] == y[valid_ids], dtype=torch.float32, device="cuda")
            for layer in LAYERS:
                values = torch.stack(cache["hidden"][layer]).float()
                train_x = values[[dev[i] for i in train_ids]].cuda()
                valid_x = values[[dev[i] for i in valid_ids]].cuda()
                vector, checkpoint = direction(train_x, train_correct, valid_x, valid_correct, layer,
                                               [rows[dev[i]]["id"] for i in train_ids])
                path = args.out / f"direction_L{layer}_F{fold}.pt"
                torch.save(checkpoint, path)
                direction_files[str(path)] = sha(path)
                train_cpu = train_x.cpu()
                valid_cpu = valid_x.cpu()
                vector = vector.cpu()
                for alpha, c in itertools.product(ALPHAS, CS):
                    scores, _ = fit_linear(steer(train_cpu, vector, alpha), train_y,
                                           steer(valid_cpu, vector, alpha), c)
                    oof[f"{layer}_{alpha}_{c}"][valid_ids] = scores
                print("SEARCH", fold, layer, flush=True)
        layer_sets = [[layer] for layer in LAYERS] + [[20, 21, 22], [21, 22, 23], [22, 23, 24]]
        rankings = []
        for layers, alpha, c, blend in itertools.product(layer_sets, ALPHAS, CS, [0.5, 0.75, 1.0]):
            score = np.mean([oof[f"{layer}_{alpha}_{c}"] for layer in layers], axis=0)
            combined = blend * score + (1 - blend) * margin / 4
            threshold = float(best_threshold(combined, y))
            pred = combined >= threshold
            metric = binary(y, pred)
            fold_gains = [100 * (np.mean(pred[valid] == y[valid]) - np.mean(base[valid] == y[valid]))
                          for _, valid in folds]
            rankings.append({"layers": layers, "alpha": alpha, "C": c, "blend": blend,
                             "threshold": threshold, "oof": metric, "fold_gains": fold_gains})
        rankings.sort(key=lambda r: (r["oof"]["accuracy"], min(r["fold_gains"]), r["oof"]["f1"],
                                     r["alpha"] != 0, -len(r["layers"])), reverse=True)
        direct = max((r for r in rankings if r["alpha"] == 0),
                     key=lambda r: (r["oof"]["accuracy"], min(r["fold_gains"]), r["oof"]["f1"]))
        aod = max((r for r in rankings if r["alpha"] != 0),
                  key=lambda r: (r["oof"]["accuracy"], min(r["fold_gains"]), r["oof"]["f1"]))
        protocol = {"model": "Qwen3-VL-4B-Instruct", "feature_sha256": source_sha,
                    "test_n": len(test), "development_n": len(dev), "layers": LAYERS,
                    "alphas": ALPHAS, "C": CS, "selection": "grouped 5-fold OOF only",
                    "test_ids_source": "Qwen2.5 v62 selected 182-question split",
                    "test_used_in_selection": False, "direction": "AOD consistency/adversarial orthogonal direction"}
        core.atomic_json(args.out / "protocol.json", protocol)
        np.savez_compressed(args.out / "oof_scores.npz", **oof)
        core.atomic_json(args.out / "selection.json", {"direct": direct, "aod": aod,
                                                        "base_oof": binary(y, base),
                                                        "direction_sha256": direction_files})
        core.atomic_json(args.out / "validation_search.json", rankings)
        print("SELECTED", json.dumps({"base": binary(y, base), "direct": direct, "aod": aod}), flush=True)
        return

    selection = json.loads((args.out / "selection.json").read_text())
    assert json.loads((args.out / "protocol.json").read_text())["feature_sha256"] == source_sha
    assert not (args.out / "results.json").exists()
    dev_y = np.array([rows[i]["gt"] for i in dev])
    dev_base = np.array([rows[i]["pred"] for i in dev])
    dev_correct = torch.tensor(dev_base == dev_y, dtype=torch.float32, device="cuda")
    test_y = np.array([rows[i]["gt"] for i in test])
    test_margin = np.array([rows[i]["margin"] for i in test])
    outputs = {}
    for kind in ("direct", "aod"):
        config = selection[kind]
        layer_scores = []
        models = []
        for layer in config["layers"]:
            values = torch.stack(cache["hidden"][layer]).float()
            train_x = values[dev].cuda()
            vector, direction_checkpoint = direction(train_x, dev_correct, train_x[:0], dev_correct[:0], layer,
                                                     [rows[i]["id"] for i in dev])
            direction_path = args.out / f"full_direction_L{layer}.pt"
            if not direction_path.exists():
                torch.save(direction_checkpoint, direction_path)
            train_cpu = train_x.cpu()
            test_cpu = values[test]
            vector = vector.cpu()
            _, payload = fit_linear(steer(train_cpu, vector, config["alpha"]), dev_y,
                                    steer(test_cpu, vector, config["alpha"]), config["C"])
            score = apply_linear(payload, steer(test_cpu, vector, config["alpha"]))
            layer_scores.append(score)
            payload.update({"layer": layer, "alpha": config["alpha"], "direction": vector,
                            "train_ids": [rows[i]["id"] for i in dev]})
            model_path = args.out / f"{kind}_L{layer}.pt"
            torch.save(payload, model_path)
            models.append({"path": str(model_path), "sha256": sha(model_path)})
        score = np.mean(layer_scores, axis=0)
        pred = predict(score, test_margin, config["blend"], config["threshold"]).astype(int)
        records = [dict(rows[idx], base_pred=rows[idx]["pred"], pred=int(p), classifier_logit=float(s))
                   for idx, p, s in zip(test, pred, score)]
        outputs[kind] = {"metrics": core.metrics(records), "records": records,
                         "models": models, "config": config}
    base_metrics = core.metrics([rows[i] for i in test])
    result = {"model": "Qwen3-VL-4B-Instruct", "n": len(test), "base": base_metrics,
              "direct": outputs["direct"]["metrics"], "aod": outputs["aod"]["metrics"],
              "direct_gain_pp": outputs["direct"]["metrics"]["accuracy"] - base_metrics["accuracy"],
              "aod_gain_pp": outputs["aod"]["metrics"]["accuracy"] - base_metrics["accuracy"]}
    core.atomic_json(args.out / "test_predictions.json", {k: v["records"] for k, v in outputs.items()})
    core.atomic_json(args.out / "frozen_models.json", {k: v["models"] for k, v in outputs.items()})
    core.atomic_json(args.out / "results.json", result)
    print("RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
