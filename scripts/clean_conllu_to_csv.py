import csv
import re

INPUT_FILES = [
    "datasets/train.conllu",
    "datasets/dev.conllu",
    "datasets/test.conllu",
    "datasets/dataset_2.conllu"
]

OUTPUT_FILE = "datasets/gondi_lexicon.csv"

# POS tags we want to keep
VALID_POS = {
    "NOUN",
    "VERB",
    "PRON",
    "ADV",
    "ADJ",
    "PROPN",
    "NUM"
}


def is_valid_word(word):
    """
    Check if token should be kept.
    """

    word = word.strip()

    # remove underscore fragments
    if word.startswith("_"):
        return False

    if re.match(r"^_+", word):
        return False

    # remove tokens containing underscore anywhere
    if "_" in word:
        return False

    # remove punctuation-only tokens
    if re.match(r"^[\.\,\-\—\–\(\)]+$", word):
        return False

    # remove single characters
    if len(word) < 2:
        return False

    # remove latin words accidentally inserted
    if re.match(r"^[a-zA-Z]+$", word):
        return False

    return True


def parse_conllu(file_path):

    rows = []

    with open(file_path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")

            if len(parts) < 4:
                continue

            word = parts[1].strip()
            pos = parts[3].strip()

            if pos not in VALID_POS:
                continue

            if not is_valid_word(word):
                continue

            rows.append((word, pos))

    return rows


def main():

    all_rows = []

    for file in INPUT_FILES:

        try:
            rows = parse_conllu(file)
            all_rows.extend(rows)
            print(f"Loaded {len(rows)} tokens from {file}")

        except FileNotFoundError:
            print(f"Skipping missing file: {file}")

    # remove duplicates
    all_rows = list(set(all_rows))

    # sort for readability
    all_rows = sorted(all_rows)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)
        writer.writerow(["word", "pos"])

        for word, pos in all_rows:
            writer.writerow([word, pos])

    print("\nDataset saved to:", OUTPUT_FILE)
    print("Total clean samples:", len(all_rows))


if __name__ == "__main__":
    main()