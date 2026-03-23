import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from training.common_training import (
    load_config,
    prepare_data_and_tokenizer,
    build_full_model,
    get_device,
    create_dataloaders
)

# -----------------------
# Load config + dataset
# -----------------------

cfg = load_config("configs/config.yaml")

tokenizer, train_ds, dev_ds, test_ds, label2id, id2label = prepare_data_and_tokenizer(cfg)

device = get_device(cfg)

# -----------------------
# Load model
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

_, test_loader = create_dataloaders(train_ds, test_ds, cfg["training"]["batch_size"])

# -----------------------
# Collect router weights
# -----------------------

router_data = {}

with torch.no_grad():

    for batch in test_loader:

        labels = batch["labels"].to(device)
        words = batch["word"]

        inputs = {k: v.to(device) for k, v in batch.items() if k not in ["labels","word"]}

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=None,
            words=words,
            return_router_weights=True
        )

        weights = outputs["router_weights"]

        for i,label_id in enumerate(labels):

            pos = id2label[label_id.item()]

            if pos not in router_data:
                router_data[pos] = []

            router_data[pos].append(weights[i].cpu().numpy())

# -----------------------
# Average weights per POS
# -----------------------

pos_tags = []
avg_weights = []

for pos in router_data:

    pos_tags.append(pos)

    avg_weights.append(
        np.mean(router_data[pos], axis=0)
    )

avg_weights = np.array(avg_weights)

languages = ["Hindi","Marathi","Telugu"]

# -----------------------
# Plot heatmap
# -----------------------

plt.figure(figsize=(6,4))

sns.heatmap(
    avg_weights,
    annot=True,
    cmap="Blues",
    xticklabels=languages,
    yticklabels=pos_tags,
    linewidths=0.5,
    linecolor="gray",
    cbar_kws={"shrink":0.8}
)

plt.title("Router Attention Across Languages")
plt.xlabel("Language Adapter")
plt.ylabel("POS Tag")

plt.tight_layout()

plt.savefig(
    "router_language_weights.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()