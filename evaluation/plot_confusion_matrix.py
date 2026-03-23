import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.common_training import (
    load_config,
    prepare_data_and_tokenizer,
    build_full_model,
    get_device,
    create_dataloaders
)

# -----------------------
# Load config
# -----------------------

cfg = load_config("configs/config.yaml")

tokenizer, train_ds, dev_ds, test_ds, label2id, id2label = prepare_data_and_tokenizer(cfg)

device = get_device(cfg)

# -----------------------
# Load trained model
# -----------------------

model = build_full_model(
    cfg,
    use_adapters=True,
    use_router=True,
    use_similarity_bias=True,
    use_lexicon_loss=True,
    label2id=label2id
)

checkpoint = torch.load(
    "outputs/full_model/checkpoint_epoch10.pt",
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# -----------------------
# Dataloader
# -----------------------

_, dev_loader = create_dataloaders(train_ds, test_ds, cfg["training"]["batch_size"])
test_loader = dev_loader
# -----------------------
# Predict
# -----------------------

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:

        labels = batch["labels"].to(device)

        inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=None
        )

        logits = outputs["logits"]

        preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
# -----------------------
# Convert IDs to labels
# -----------------------

true_tags = [id2label[i] for i in all_labels]
pred_tags = [id2label[i] for i in all_preds]

# -----------------------
# Keep only selected tags
# -----------------------

target_tags = ["NOUN", "VERB", "ADV", "PRON"]

filtered_true = []
filtered_pred = []

for t, p in zip(true_tags, pred_tags):
    if t in target_tags:
        filtered_true.append(t)
        filtered_pred.append(p)

# -----------------------
# Confusion Matrix
# -----------------------

cm = confusion_matrix(filtered_true, filtered_pred, labels=target_tags)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_tags,
    yticklabels=target_tags,
    linewidths=0.5,
    linecolor="gray",
    square=True,
    cbar_kws={"shrink":0.8}
)

plt.xlabel("Predicted POS Tag")
plt.ylabel("True POS Tag")
plt.title("Confusion Matrix (Major POS Tags)")

plt.tight_layout()

plt.savefig(
    "pos_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()