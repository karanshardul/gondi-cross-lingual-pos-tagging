import pandas as pd
import random

INPUT = "datasets/gondi_lexicon.csv"
OUTPUT = "datasets/gondi_lexicon_augmented.csv"

def add_noise(word):

    if len(word) < 4:
        return word

    i = random.randint(1, len(word)-2)

    return word[:i] + word[i+1:]


df = pd.read_csv(INPUT)

augmented = []

for _, row in df.iterrows():

    word = row["word"]
    pos = row["pos"]

    augmented.append((word, pos))

    noisy = add_noise(word)

    augmented.append((noisy, pos))


df2 = pd.DataFrame(augmented, columns=["word","pos"])

df2.to_csv(OUTPUT, index=False)

print("Original:", len(df))
print("Augmented:", len(df2))