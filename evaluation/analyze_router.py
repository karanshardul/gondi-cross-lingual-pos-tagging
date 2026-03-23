import argparse
import json
import os
from typing import Dict, Any, List

import numpy as np
import torch
import matplotlib.pyplot as plt

from training.common_training import (
    load_config,
    prepare_data_and_tokenizer,
    build_full_model,
    get_device,
)


LANGS = ["hi", "mr", "te"]


def _load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


def analyze_router(config_path: str, experiment: str, checkpoint_path: str, split: str = "test") -> str:
    if experiment not in ("router", "full_model"):
        raise ValueError("Router analysis is only valid for 'router' and 'full_model' experiments.")

    cfg = load_config(config_path)
    tokenizer, train_ds, dev_ds, test_ds, label2id, id2label = prepare_data_and_tokenizer(cfg)
    device = get_device(cfg)

    flags = dict(
        use_adapters=True,
        use_router=True,
        use_similarity_bias=(experiment == "full_model"),
        use_lexicon_loss=(experiment == "full_model"),
    )
    model = build_full_model(cfg, label2id=label2id, **flags)
    model.to(device)
    ckpt = _load_checkpoint(model, checkpoint_path, device)
    model.eval()

    ds = {"train": train_ds, "dev": dev_ds, "test": test_ds}[split]
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=lambda batch: {
            "input_ids": torch.stack([b["input_ids"] for b in batch], 0),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch], 0),
            "labels": torch.stack([b["labels"] for b in batch], 0),
            "word": [b["word"] for b in batch],
        },
    )

    weights_all: List[np.ndarray] = []
    words_all: List[str] = []

    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=None,
                words=batch["word"],
                lambda_lex=0.0,
                label2id=label2id,
                lexicon=None,
                return_router_weights=True,
            )
            w = outputs["router_weights"]  # [batch, 3]
            if w is None:
                continue
            weights_all.append(w.cpu().numpy())
            words_all.extend(batch["word"])

    if not weights_all:
        raise RuntimeError("No router weights produced. Check that the checkpoint uses a router model.")

    W = np.concatenate(weights_all, axis=0)  # [N, 3]
    mean_usage = W.mean(axis=0).tolist()

    out_dir = os.path.join(cfg["experiment"]["output_dir"], experiment)
    os.makedirs(out_dir, exist_ok=True)

    # Save per-word weights (large but useful for analysis)
    weights_path = os.path.join(out_dir, f"router_weights_{split}_epoch{int(ckpt.get('epoch', -1))}.json")
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": experiment,
                "split": split,
                "checkpoint_path": checkpoint_path,
                "epoch": int(ckpt.get("epoch", -1)),
                "languages": LANGS,
                "mean_adapter_usage": {lang: mean_usage[i] for i, lang in enumerate(LANGS)},
                "weights": [{"word": words_all[i], "hi": float(W[i, 0]), "mr": float(W[i, 1]), "te": float(W[i, 2])} for i in range(len(words_all))],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Heatmap: sample up to 100 words for readability
    n_show = min(100, W.shape[0])
    W_show = W[:n_show, :]
    words_show = words_all[:n_show]

    fig, ax = plt.subplots(figsize=(8, max(4, n_show * 0.12)))
    im = ax.imshow(W_show, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xticks(range(len(LANGS)))
    ax.set_xticklabels(LANGS)
    ax.set_yticks(range(n_show))
    ax.set_yticklabels(words_show, fontsize=7)
    ax.set_title(f"Router Weights Heatmap ({experiment}, {split})")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Adapter Weight")
    fig.tight_layout()

    plot_path = os.path.join(out_dir, f"router_heatmap_{split}_epoch{int(ckpt.get('epoch', -1))}.png")
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    summary_path = os.path.join(out_dir, f"router_mean_usage_{split}_epoch{int(ckpt.get('epoch', -1))}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": experiment,
                "split": split,
                "epoch": int(ckpt.get("epoch", -1)),
                "languages": LANGS,
                "mean_adapter_usage": {lang: mean_usage[i] for i, lang in enumerate(LANGS)},
                "heatmap_path": plot_path,
                "weights_path": weights_path,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--experiment", type=str, required=True, choices=["router", "full_model"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"])
    args = parser.parse_args()

    out = analyze_router(args.config, args.experiment, args.checkpoint, split=args.split)
    print(f"Saved router analysis to {out}")

