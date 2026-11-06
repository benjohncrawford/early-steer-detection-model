import random
from pathlib import Path
import shutil
import json

# Generate text file describing test and train
def generate_split(train_split, train_filepath, test_filepath, label_json_filepath):
    correct_ids = []
    incorrect_ids = []
    # Load the review JSON and map each sample id to 1 (correct) or 0 (incorrect/uncertain)
    with open(label_json_filepath, "r", encoding="utf-8") as f:
        reviews = json.load(f)
        for entry in reviews:
            sample_id = entry["id"]
            if entry["status"] == "correct":
                correct_ids.append(sample_id)
            else:
                incorrect_ids.append(sample_id)
    
    
    random.shuffle(correct_ids)
    random.shuffle(incorrect_ids)
    train_indexes = correct_ids[:int(len(correct_ids)*train_split)] + incorrect_ids[:int(len(incorrect_ids)*train_split)]
    test_indexes = correct_ids[int(len(correct_ids)*train_split):] + incorrect_ids[int(len(incorrect_ids)*train_split):]
    train_indexes.sort()
    test_indexes.sort()

    with open(train_filepath, "w+") as f:
        for index in train_indexes:
            f.write(index + "\n")

    with open(test_filepath, "w+") as f:
        for index in test_indexes:
            f.write(index + "\n")
    

def read_indexes(train_filepath, test_filepath):
    train_indexes = []
    test_indexes = []
    with open(train_filepath, "r") as f:
        for index in f.readlines():
            train_indexes.append(f"{int(index):04d}")
    with open(test_filepath, "r") as f:
        for index in f.readlines():
            test_indexes.append(f"{int(index):04d}")  
    return train_indexes, test_indexes

# Move files
def move_files(train_indexes, test_indexes, source_directory, train_destination, test_destination):
        source_directory = Path(source_directory)
        train_destination = Path(train_destination)
        test_destination = Path(test_destination)
        
        # Create the destinations if they don't exist
        train_destination.mkdir(parents=True, exist_ok=True)
        test_destination.mkdir(parents=True, exist_ok=True)
        
        # get all safetensor files in the directory and any subdirectories
        tensor_files = [f for f in source_directory.rglob("*.safetensors") if f.is_file()]
        image_files = [f for f in source_directory.rglob("*.png") if f.is_file()]
        json_files = [f for f in source_directory.rglob("*.json") if f.is_file()]
        
        def iterate_files_and_move_to_correct_destination(files):
            for file in files:
                index = str(file).split("\\")[-1].split(".")[0]
                if index in train_indexes:
                    shutil.move(str(file), str(train_destination))
                else:
                    shutil.move(str(file), str(test_destination))

        
        iterate_files_and_move_to_correct_destination(tensor_files)
        iterate_files_and_move_to_correct_destination(image_files)
        iterate_files_and_move_to_correct_destination(json_files)

            

train_split = 0.8
train_txt = "train_indexes.txt" 
test_txt = "test_indexes.txt"
label_json_filepath = "C:\\Users\\ben99\\Downloads\\sd3-numeracy-v2\\Results\\sd3-numeracy-v2-backpacks-reviews.json"
generate_split(train_split, train_txt, test_txt, label_json_filepath)


train_indexes, test_indexes = read_indexes(train_txt, test_txt)

source_dir = "C:\\Users\\ben99\\Downloads\\sd3-numeracy-v2\\Results"
train_destination = "C:\\Users\\ben99\\Downloads\\sd3-numeracy-v2\\Results\\Train"
test_destination = "C:\\Users\\ben99\\Downloads\\sd3-numeracy-v2\\Results\\Test"
move_files(train_indexes, test_indexes, source_dir, train_destination, test_destination)