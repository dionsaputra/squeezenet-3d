import csv
import os
from jester import JesterDataset
import json
import shutil

jester_data_path = "/home/ds/Data/academic/ta/jester/20bn-jester-v1"
train_csv_path = "/home/ds/Data/academic/ta/jester/jester-v1-train.csv"
val_csv_path = "/home/ds/Data/academic/ta/jester/jester-v1-validation.csv"
dest = "/home/ds/Data/academic/jester_new_mini"

selected_labels = [
    JesterDataset.Label.swiping_left,
    JesterDataset.Label.swiping_right,
    JesterDataset.Label.swiping_down,
    JesterDataset.Label.swiping_up,
    JesterDataset.Label.pushing_two_fingers_away,
    JesterDataset.Label.pulling_two_fingers_in,
    JesterDataset.Label.turning_hand_clockwise,
    JesterDataset.Label.turning_hand_counterclockwise,
    JesterDataset.Label.zooming_in_with_full_hand,
    JesterDataset.Label.zooming_out_with_full_hand,
    JesterDataset.Label.thumb_up,
    JesterDataset.Label.thumb_down,
    JesterDataset.Label.stop_sign,
    JesterDataset.Label.no_gesture,
    JesterDataset.Label.doing_other_things,
]


def save_path_to_json(selected_labels, csv_path, max_item_per_label, json_path):
    items = JesterDataset.Item.get_items_from_csv(
        csv_path, selected_labels, max_item_per_label
    )
    paths = []
    for item in items:
        path = os.path.join(jester_data_path, item.id)
        paths.append(path)
    with open(json_path, "w") as file:
        json.dump(paths, file)


def copy_directory(json_path_file, destination):
    with open(json_path_file) as file:
        paths = json.load(file)
        counter, total = 0, len(paths)
        for path in paths:
            dir_name = path.split("/")[-1]
            os.system(f"cp -rf {path} {destination}/{dir_name}")
            counter += 1
            print(f"copying {counter}/{total}")
            # print(f"Copy directory {dir_name}")


def create_csv_annotation(json_file, source_file, destination_file):
    with open(json_file) as jsonfile:
        ids = json.load(jsonfile)
        selected_ids = []
        for id in ids:
            selected_ids.append(id.split("/")[-1])

    with open(source_file) as file:
        selected_row = []
        for row in csv.reader(file):
            item = JesterDataset.Item.parse(row[0])
            if item.id in selected_ids:
                selected_row.append(row)

    with open(destination_file, 'w') as file:
        writer = csv.writer(file)
        for row in selected_row:
            writer.writerow(row)


if __name__ == "__main__":
    save_path_to_json(selected_labels, train_csv_path, 10, "mini_train.json")
    save_path_to_json(selected_labels, val_csv_path, 5, "mini_val.json")
    copy_directory("mini_train.json", dest)
    copy_directory("mini_val.json", dest)
    create_csv_annotation("mini_train.json", train_csv_path, "mini_train.csv")
    create_csv_annotation("mini_val.json", val_csv_path, "mini_val.csv")
