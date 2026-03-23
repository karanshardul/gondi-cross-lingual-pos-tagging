import csv

INPUT_FILES = [
    "datasets/train.conllu",
    "datasets/dev.conllu",
    "datasets/test.conllu",
    "datasets/dataset_2.conllu"
]

OUTPUT_FILE = "datasets/gondi_lexicon.csv"


def parse_conllu(file_path):
    pairs = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")

            if len(parts) < 4:
                continue

            word = parts[1]
            pos = parts[3]

            if word == "_" or pos == "_":
                continue

            pairs.append((word, pos))

    return pairs


def main():
    all_pairs = []

    for file in INPUT_FILES:
        all_pairs.extend(parse_conllu(file))

    # remove duplicates
    all_pairs = list(set(all_pairs))

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "pos"])

        for word, pos in all_pairs:
            writer.writerow([word, pos])

    print("Saved:", OUTPUT_FILE)
    print("Total samples:", len(all_pairs))


if __name__ == "__main__":
    main()