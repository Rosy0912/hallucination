"""Select category-aware confidence gates on development OOF, then test once."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

import run as core
from qwen3vl4b_hb_aod import SEED, apply_linear, fit_linear, predict, sha
from qwen_layerstudy_v28 import binary, direction, steer


def best_gated(score, labels, base, margin, mask):
    best = None
    for gate in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 1e6]:
        active = mask & (np.abs(margin) <= gate)
        inactive_correct = int(np.sum((base == labels) & mask & ~active))
        indices = np.flatnonzero(active)
        if not len(indices):
            candidates = [(inactive_correct, 0.0)]
        else:
            order = indices[np.argsort(-score[indices])]
            active_correct = int(np.sum(labels[order] == 0))
            candidates = [(inactive_correct + active_correct, float(score[order[0]] + 1))]
            position = 0
            while position < len(order):
                end = position + 1
                while end < len(order) and score[order[end]] == score[order[position]]:
                    end += 1
                group = labels[order[position:end]]
                active_correct += int(np.sum(group == 1) - np.sum(group == 0))
                next_score = score[order[end]] if end < len(order) else score[order[position]] - 1
                threshold = float((score[order[position]] + next_score) / 2)
                candidates.append((inactive_correct + active_correct, threshold))
                position = end
        for correct, threshold in candidates:
            accuracy = correct / int(mask.sum())
            candidate = (accuracy, -gate, -abs(float(threshold)), float(gate), float(threshold))
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("select", "test"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    feature_path = core.ROOT / "results/qwen3vl4b_hb/features.pt"
    cache = torch.load(feature_path, map_location="cpu", weights_only=False)
    rows = cache["records"]
    test_ids = {r["id"] for r in json.loads((core.ROOT / "results/qwen_resplit_v62/fold1/test_predictions.json").read_text())}
    dev = [i for i, r in enumerate(rows) if r["id"] not in test_ids]
    test = [i for i, r in enumerate(rows) if r["id"] in test_ids]
    y = np.array([rows[i]["gt"] for i in dev])
    base = np.array([rows[i]["pred"] for i in dev])
    margin = np.array([rows[i]["margin"] for i in dev])
    categories = np.array([rows[i]["raw"]["category"] for i in dev])
    source = args.out.parent / "qwen3vl4b_hb_aod"

    if args.stage == "select":
        assert not (args.out / "selection.json").exists()
        args.out.mkdir(parents=True, exist_ok=True)
        scores = np.load(source / "oof_scores.npz")
        rankings = json.loads((source / "validation_search.json").read_text())
        chosen = {}
        for category in sorted(set(categories)):
            mask = categories == category
            candidates = []
            for config in rankings:
                layer_score = np.mean([scores[f"{layer}_{config['alpha']}_{config['C']}"]
                                       for layer in config["layers"]], axis=0)
                combined = config["blend"] * layer_score + (1 - config["blend"]) * margin / 4
                accuracy, _, _, gate, threshold = best_gated(combined, y, base, margin, mask)
                candidates.append(dict(layers=config["layers"], alpha=config["alpha"], C=config["C"],
                                       blend=config["blend"], gate=gate, threshold=threshold,
                                       category=category, oof_category_accuracy=100 * accuracy))
            candidates.sort(key=lambda r: (r["oof_category_accuracy"], r["alpha"] != 0,
                                           -len(r["layers"]), -r["gate"]), reverse=True)
            chosen[category] = candidates[0]
        prediction = base.copy()
        for category, config in chosen.items():
            mask = categories == category
            layer_score = np.mean([scores[f"{layer}_{config['alpha']}_{config['C']}"]
                                   for layer in config["layers"]], axis=0)
            combined = config["blend"] * layer_score + (1 - config["blend"]) * margin / 4
            active = mask & (np.abs(margin) <= config["gate"])
            prediction[active] = combined[active] >= config["threshold"]
        result = {"model": "Qwen3-VL-4B-Instruct", "configurations": chosen,
                  "base_oof": binary(y, base), "selected_oof": binary(y, prediction),
                  "gain_pp": binary(y, prediction)["accuracy"] - binary(y, base)["accuracy"],
                  "selection_source": "development grouped OOF only", "test_used": False,
                  "source_sha256": sha(source / "oof_scores.npz")}
        core.atomic_json(args.out / "selection.json", result)
        print("HARDCASE SELECTED", json.dumps(result), flush=True)
        return

    selection = json.loads((args.out / "selection.json").read_text())
    assert selection["source_sha256"] == sha(source / "oof_scores.npz")
    assert not (args.out / "results.json").exists()
    dev_y = np.array([rows[i]["gt"] for i in dev])
    dev_base = np.array([rows[i]["pred"] for i in dev])
    dev_correct = torch.tensor(dev_base == dev_y, dtype=torch.float32, device="cuda")
    test_y = np.array([rows[i]["gt"] for i in test])
    test_base = np.array([rows[i]["pred"] for i in test])
    test_margin = np.array([rows[i]["margin"] for i in test])
    test_categories = np.array([rows[i]["raw"]["category"] for i in test])
    final = test_base.copy()
    model_files = []
    diagnostics = {}
    for category, config in selection["configurations"].items():
        layer_scores = []
        for layer in config["layers"]:
            values = torch.stack(cache["hidden"][layer]).float()
            train_x = values[dev].cuda()
            vector, direction_checkpoint = direction(train_x, dev_correct, train_x[:0], dev_correct[:0], layer,
                                                     [rows[i]["id"] for i in dev])
            direction_path = args.out / f"direction_{category}_L{layer}.pt"
            torch.save(direction_checkpoint, direction_path)
            train_cpu, test_cpu, vector = train_x.cpu(), values[test], vector.cpu()
            _, payload = fit_linear(steer(train_cpu, vector, config["alpha"]), dev_y,
                                    steer(test_cpu, vector, config["alpha"]), config["C"])
            score = apply_linear(payload, steer(test_cpu, vector, config["alpha"]))
            layer_scores.append(score)
            payload.update(layer=layer, alpha=config["alpha"], direction=vector,
                           train_ids=[rows[i]["id"] for i in dev])
            model_path = args.out / f"head_{category}_L{layer}.pt"
            torch.save(payload, model_path)
            model_files.append({"path": str(model_path), "sha256": sha(model_path)})
        score = np.mean(layer_scores, axis=0)
        combined = config["blend"] * score + (1 - config["blend"]) * test_margin / 4
        category_mask = test_categories == category
        active = category_mask & (np.abs(test_margin) <= config["gate"])
        final[active] = combined[active] >= config["threshold"]
        diagnostics[category] = {"n": int(category_mask.sum()), "active": int(active.sum())}
    records = [dict(rows[index], base_pred=int(base_pred), pred=int(pred))
               for index, base_pred, pred in zip(test, test_base, final)]
    base_metrics = core.metrics([rows[i] for i in test])
    selected_metrics = core.metrics(records)
    result = {"model": "Qwen3-VL-4B-Instruct", "n": len(test), "base": base_metrics,
              "aod": selected_metrics, "gain_pp": selected_metrics["accuracy"] - base_metrics["accuracy"],
              "diagnostics": diagnostics, "selection_sha256": sha(args.out / "selection.json"),
              "models": model_files}
    core.atomic_json(args.out / "test_predictions.json", records)
    core.atomic_json(args.out / "results.json", result)
    print("HARDCASE RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
