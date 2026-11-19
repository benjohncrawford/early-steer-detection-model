import random
from pathlib import Path
import shutil
import json


# Generate text file describing test and train
def generate_split(train_split, train_filepath, test_filepath, label_json_filepath, seed=42):
    correct_ids = []
    incorrect_ids = []
    # Load the review JSON and map each sample id to 1 (correct) or 0 (incorrect/uncertain)
    with open(label_json_filepath, "r", encoding="utf-8") as f:
        reviews = json.load(f)
        for entry in reviews:
            sample_id = entry["id"]
            if entry["annotation_status"] == "correct":
                correct_ids.append(sample_id)
            else:
                incorrect_ids.append(sample_id)

    rng = random.Random(seed)
    rng.shuffle(correct_ids)
    rng.shuffle(incorrect_ids)

    n_train_correct = int(len(correct_ids) * train_split)
    n_train_incorrect = int(len(incorrect_ids) * train_split)

    train_indexes = correct_ids[:n_train_correct] + incorrect_ids[:n_train_incorrect]
    test_indexes = correct_ids[n_train_correct:] + incorrect_ids[n_train_incorrect:]
    train_indexes.sort()
    test_indexes.sort()

    with open(train_filepath, "w", encoding="utf-8") as f:
        for index in train_indexes:
            f.write(index + "\n")

    with open(test_filepath, "w", encoding="utf-8") as f:
        for index in test_indexes:
            f.write(index + "\n")


def read_indexes(train_filepath, test_filepath):
    train_indexes = []
    test_indexes = []
    with open(train_filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                train_indexes.append(f"{int(stripped):04d}")
    with open(test_filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                test_indexes.append(f"{int(stripped):04d}")
    return train_indexes, test_indexes


# Move files
def move_files(train_indexes, test_indexes, source_directory, train_destination, test_destination):
    source_directory = Path(source_directory)
    train_destination = Path(train_destination)
    test_destination = Path(test_destination)

    # Create the destinations if they don't exist
    train_destination.mkdir(parents=True, exist_ok=True)
    test_destination.mkdir(parents=True, exist_ok=True)

    train_set = set(train_indexes)
    test_set = set(test_indexes)

    # Get all safetensor, image, and metadata json files in the directory and subdirectories
    tensor_files = [f for f in source_directory.rglob("*.safetensors") if f.is_file()]
    image_files = [f for f in source_directory.rglob("*.png") if f.is_file()]
    json_files = [f for f in source_directory.rglob("*.json") if f.is_file()]

    def iterate_files_and_move_to_correct_destination(files):
        for file in files:
            index = file.stem
            if index in train_set:
                shutil.move(str(file), str(train_destination))
            elif index in test_set:
                shutil.move(str(file), str(test_destination))

    iterate_files_and_move_to_correct_destination(tensor_files)
    iterate_files_and_move_to_correct_destination(image_files)
    iterate_files_and_move_to_correct_destination(json_files)


if __name__ == "__main__":
    train_split = 0.8
    train_txt = "train_indexes.txt"
    test_txt = "test_indexes.txt"
    label_json_filepath = "D:\\sd35-numeracy-v3-share\\annotations.json"
    generate_split(train_split, train_txt, test_txt, label_json_filepath)

    train_indexes, test_indexes = read_indexes(train_txt, test_txt)

    source_dir = "D:\\sd35-numeracy-v3-share\\Results"
    train_destination = "D:\\sd35-numeracy-v3-share\\Train"
    test_destination = "D:\\sd35-numeracy-v3-share\\Test"
    move_files(train_indexes, test_indexes, source_dir, train_destination, test_destination)