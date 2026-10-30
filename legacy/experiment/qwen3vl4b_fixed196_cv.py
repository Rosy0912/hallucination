"""Five-fold grouped AOD ensemble on the original fixed 196-question test split."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

import run as core
from qwen3vl4b_hb_aod import apply_linear, fit_linear, sha
from qwen3vl4b_hb_hardcase import best_gated
from qwen_layerstudy_v28 import binary, direction, steer


CONFIGS = {
    "VD": {"layer": 21, "alpha": -0.5, "C": 0.001},
    "VS": {"layer": 24, "alpha": -0.5, "C": 0.01},
}
SEED = 20260929


def data():
    feature_path = core.ROOT / "results/qwen3vl4b_hb/features.pt"
    cache = torch.load(feature_path, map_location="cpu", weights_only=False)
    rows = cache["records"]
    dev = [i for i, row in enumerate(rows) if row["split"] in ("train", "val")]
    test = [i for i, row in enumerate(rows) if row["split"] == "test"]
    assert len(dev) == 755 and len(test) == 196
    labels = np.array([rows[i]["gt"] for i in dev])
    strata = np.array([f"{rows[i]['raw']['category']}:{rows[i]['gt']}" for i in dev])
    groups = np.array([rows[i]["group"] for i in dev])
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
                 .split(np.zeros(len(dev)), strata, groups))
    return feature_path, cache, rows, dev, test, labels, groups, folds


def fold_stage(args):
    feature_path, cache, rows, dev, test, labels, groups, folds = data()
    train_local, held_local = folds[args.fold]
    assert not set(groups[train_local]) & set(groups[held_local])
    train = [dev[i] for i in train_local]
    held = [dev[i] for i in held_local]
    train_y = labels[train_local]
    train_base = np.array([rows[i]["pred"] for i in train])
    held_base = np.array([rows[i]["pred"] for i in held])
    train_correct = torch.tensor(train_base == train_y, dtype=torch.float32, device="cuda")
    held_correct = torch.tensor(held_base == labels[held_local], dtype=torch.float32, device="cuda")
    folder = args.out / f"fold{args.fold}"
    assert not folder.exists()
    folder.mkdir(parents=True)
    arrays = {"held_local": held_local}
    model_files = []
    for category, config in CONFIGS.items():
        values = torch.stack(cache["hidden"][config["layer"]]).float()
        vector, checkpoint = direction(values[train].cuda(), train_correct,
                                       values[held].cuda(), held_correct, config["layer"],
                                       [rows[i]["id"] for i in train])
        vector = vector.cpu()
        direction_path = folder / f"direction_{category}.pt"
        torch.save(checkpoint, direction_path)
        held_score, payload = fit_linear(steer(values[train], vector, config["alpha"]), train_y,
                                         steer(values[held], vector, config["alpha"]), config["C"])
        test_score = apply_linear(payload, steer(values[test], vector, config["alpha"]))
        arrays[f"held_{category}"] = held_score
        arrays[f"test_{category}"] = test_score
        payload.update(config=config, direction=vector, train_ids=[rows[i]["id"] for i in train])
        model_path = folder / f"head_{category}.pt"
        torch.save(payload, model_path)
        model_files.append({"path": str(model_path), "sha256": sha(model_path)})
    np.savez_compressed(folder / "scores.npz", **arrays)
    core.atomic_json(folder / "manifest.json", {
        "fold": args.fold, "seed": SEED, "feature_sha256": sha(feature_path),
        "train_ids": [rows[i]["id"] for i in train], "held_ids": [rows[i]["id"] for i in held],
        "test_ids": [rows[i]["id"] for i in test], "models": model_files,
        "test_used_for_selection": False,
    })
    print("CV FOLD COMPLETE", args.fold, len(train), len(held), flush=True)


def summary_stage(args):
    feature_path, cache, rows, dev, test, labels, groups, folds = data()
    oof = {category: np.zeros(len(dev)) for category in CONFIGS}
    test_scores = {category: [] for category in CONFIGS}
    manifests = []
    for fold in range(5):
        folder = args.out / f"fold{fold}"
        manifest = json.loads((folder / "manifest.json").read_text())
        assert manifest["feature_sha256"] == sha(feature_path)
        scores = np.load(folder / "scores.npz")
        held_local = scores["held_local"]
        assert set(manifest["held_ids"]) == {rows[dev[int(i)]]["id"] for i in held_local}
        for category in CONFIGS:
            oof[category][held_local] = scores[f"held_{category}"]
            test_scores[category].append(scores[f"test_{category}"])
        manifests.append(manifest)
    assert sorted(i for _, held in folds for i in held) == list(range(len(dev)))
    dev_base = np.array([rows[i]["pred"] for i in dev])
    dev_margin = np.array([rows[i]["margin"] for i in dev])
    dev_category = np.array([rows[i]["raw"]["category"] for i in dev])
    choices = {}
    validation_prediction = dev_base.copy()
    for category in CONFIGS:
        mask = dev_category == category
        candidates = []
        for blend in [0.5, 0.75, 1.0]:
            combined = blend * oof[category] + (1 - blend) * dev_margin / 4
            accuracy, _, _, gate, threshold = best_gated(
                combined, labels, dev_base, dev_margin, mask)
            candidates.append((accuracy, -gate, -abs(threshold), blend, gate, threshold))
        _, _, _, blend, gate, threshold = max(candidates)
        choices[category] = dict(CONFIGS[category], blend=blend, gate=gate, threshold=threshold)
        combined = blend * oof[category] + (1 - blend) * dev_margin / 4
        active = mask & (np.abs(dev_margin) <= gate)
        validation_prediction[active] = combined[active] >= threshold

    test_base = np.array([rows[i]["pred"] for i in test])
    test_margin = np.array([rows[i]["margin"] for i in test])
    test_category = np.array([rows[i]["raw"]["category"] for i in test])
    final = test_base.copy()
    for category, config in choices.items():
        score = np.mean(test_scores[category], axis=0)
        combined = config["blend"] * score + (1 - config["blend"]) * test_margin / 4
        active = (test_category == category) & (np.abs(test_margin) <= config["gate"])
        final[active] = combined[active] >= config["threshold"]
    records = [dict(rows[index], base_pred=int(base), pred=int(pred))
               for index, base, pred in zip(test, test_base, final)]
    base_metrics = core.metrics([rows[i] for i in test])
    selected_metrics = core.metrics(records)
    result = {"model": "Qwen3-VL-4B-Instruct", "scope": "original fixed image-only split",
              "counts": {"development": 755, "test": 196}, "configurations": choices,
              "base_development_oof": binary(labels, dev_base),
              "aod_development_oof": binary(labels, validation_prediction),
              "base": base_metrics, "aod": selected_metrics,
              "gain_pp": selected_metrics["accuracy"] - base_metrics["accuracy"],
              "fold_manifests": [sha(args.out / f"fold{i}/manifest.json") for i in range(5)],
              "test_used_for_selection": False}
    core.atomic_json(args.out / "test_predictions.json", records)
    core.atomic_json(args.out / "results.json", result)
    print("FIXED196 CV RESULT", json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("fold", "summary"), required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "fold":
        assert args.fold in range(5)
        fold_stage(args)
    else:
        summary_stage(args)


if __name__ == "__main__":
    main()
