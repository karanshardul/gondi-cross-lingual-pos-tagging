import argparse
import os

import torch
from sklearn.metrics import accuracy_score, f1_score

from training.common_training import (
    load_config,
    prepare_data_and_tokenizer,
    build_full_model,
    create_dataloaders,
    create_optimizer_and_scheduler,
    set_seed,
    get_device,
    save_checkpoint,
    setup_logging,
    save_metrics_json,
)


def move_batch_to_device(batch, device):
    """
    Move only tensor values to the device.
    Ignore metadata like strings or lists (e.g., 'word').
    """
    return {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}


def train_baseline(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["experiment"]["seed"])

    tokenizer, train_ds, dev_ds, test_ds, label2id, id2label = prepare_data_and_tokenizer(cfg)

    device = get_device(cfg)
    logger = setup_logging(cfg["experiment"]["output_dir"], "baseline")

    logger.info(f"Using device: {device}")
    logger.info(f"Train size: {len(train_ds)} | Dev size: {len(dev_ds)} | Test size: {len(test_ds)}")

    model = build_full_model(
        cfg,
        use_adapters=False,
        use_router=False,
        use_similarity_bias=False,
        use_lexicon_loss=False,
        label2id=label2id,
    )

    model.to(device)

    train_loader, dev_loader = create_dataloaders(
        train_ds,
        dev_ds,
        cfg["training"]["batch_size"],
    )

    num_training_steps = len(train_loader) * cfg["training"]["num_epochs"]

    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        cfg,
        num_training_steps,
    )

    for epoch in range(1, cfg["training"]["num_epochs"] + 1):

        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):

            batch = move_batch_to_device(batch, device)

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                words=None,
                lambda_lex=0.0,
                label2id=label2id,
                lexicon=None,
            )

            loss = outputs["loss"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg["training"]["gradient_clip"],
            )

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)

        logger.info(f"[Baseline] Epoch {epoch} - train loss: {avg_loss:.4f}")

        # ========================
        # Dev evaluation
        # ========================

        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():

            for batch in dev_loader:

                labels = batch["labels"].to(device)

                inputs = move_batch_to_device(batch, device)

                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=None,
                    words=None,
                    lambda_lex=0.0,
                    label2id=label2id,
                    lexicon=None,
                )

                logits = outputs["logits"]

                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro")

        logger.info(
            f"[Baseline] Epoch {epoch} - dev accuracy: {acc:.4f} | macro F1: {macro_f1:.4f}"
        )

        save_metrics_json(
            cfg,
            "baseline",
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "dev_accuracy": acc,
                "dev_macro_f1": macro_f1,
            },
            filename=f"metrics_epoch{epoch}.json",
        )

        if cfg["training"]["save_every_epoch"]:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                cfg,
                experiment_name="baseline",
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")

    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    train_baseline(args.config)