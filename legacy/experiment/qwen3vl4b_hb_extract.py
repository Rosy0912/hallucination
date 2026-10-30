"""Extract Qwen3-VL-4B HallusionBench last-prompt states at layers 16--24."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import run as core


MODEL_NAME = "Qwen3-VL-4B-Instruct"
LAYERS = list(range(16, 25))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    source = core.ROOT / "results/qwen_hb_imageonly/features.pt"
    rows = torch.load(source, map_location="cpu", weights_only=False)["records"]
    assert len(rows) == 951 and all(r["image"] for r in rows)
    split_groups = {s: {r["group"] for r in rows if r["split"] == s} for s in ("train", "val", "test")}
    assert all(not split_groups[a] & split_groups[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test")))

    cache_path = args.out / "features.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False) if cache_path.exists() else {
        "records": [], "hidden": {layer: [] for layer in LAYERS}
    }
    done = {r["id"] for r in cache["records"]}
    assert len(done) == len(cache["records"])

    loaded = core.load_model(SimpleNamespace(
        model=str(core.ROOT / "models" / MODEL_NAME), max_pixels=1003520
    ))
    yes_ids, no_ids = core.get_yes_no_token_ids(loaded.tokenizer)
    blocks, block_path = core.resolve_decoder_layers(loaded.model)
    assert max(LAYERS) <= len(blocks)
    captured = {}
    handles = []
    for layer in LAYERS:
        def hook(_module, _inputs, output, layer=layer):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer] = hidden[0, -1].detach().float().cpu().clone()
        handles.append(blocks[layer - 1].register_forward_hook(hook))

    protocol = {
        "model": MODEL_NAME,
        "model_class": type(loaded.model).__name__,
        "layers": LAYERS,
        "token": "last prompt token before answer generation",
        "block_path": block_path,
        "source_sha256": sha(source),
        "code_sha256": sha(__file__),
        "max_pixels": 1003520,
        "actual_images": True,
        "job": os.environ.get("SLURM_JOB_ID"),
        "scoring": "max Yes variants >= max No variants",
    }
    core.atomic_json(args.out / "protocol.json", protocol)

    def save():
        temporary = cache_path.with_suffix(".tmp")
        torch.save(cache, temporary)
        temporary.replace(cache_path)
        core.atomic_json(args.out / "base_metrics.json", core.summary(cache["records"]))

    started = time.time()
    try:
        with torch.inference_mode():
            for row in rows:
                if row["id"] in done:
                    continue
                captured.clear()
                inputs = core.inputs_for(loaded, row)
                assert inputs["pixel_values"].numel() > 0
                logits = core.forward_next_logits(loaded.model, inputs).float()
                margin = float(logits[0, yes_ids].max() - logits[0, no_ids].max())
                record = dict(row, pred=int(margin >= 0), margin=margin,
                              greedy_token=loaded.tokenizer.decode([int(logits.argmax(-1).item())]))
                cache["records"].append(record)
                for layer in LAYERS:
                    cache["hidden"][layer].append(captured[layer])
                if len(cache["records"]) % 25 == 0:
                    save()
                    print("EXTRACTED", len(cache["records"]), "OF", len(rows),
                          "SECONDS", round(time.time() - started, 1), flush=True)
        save()
    finally:
        for handle in handles:
            handle.remove()
    protocol["feature_sha256"] = sha(cache_path)
    protocol["n"] = len(cache["records"])
    core.atomic_json(args.out / "protocol.json", protocol)
    print(json.dumps(core.summary(cache["records"]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
