import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("datasets/gondi_lexicon_augmented.csv")

# Count POS tags
pos_counts = df["pos"].value_counts().sort_values(ascending=False)

plt.figure(figsize=(8,5))

pos_counts.plot(
    kind="bar",
    color="steelblue",
    edgecolor="black"
)

plt.title("POS Tag Distribution in Gondi Dataset", fontsize=13)
plt.xlabel("POS Tag", fontsize=12)
plt.ylabel("Frequency", fontsize=12)

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "pos_tag_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()