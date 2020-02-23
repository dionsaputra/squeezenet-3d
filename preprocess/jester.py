import csv
import torch
import os
import glob
import numpy
from PIL import Image
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms


class JesterDataset(Dataset):
    def __init__(
        self,
        train_dir,
        train_csv_path,
        selected_labels,
        max_item_per_label,
        is_train=True,
        transform=None,
        count_frame=16,
    ):
        self.train_dir = train_dir
        self.train_items = JesterDataset.Item.get_items_from_csv(
            train_csv_path, selected_labels, max_item_per_label
        )
        self.is_train = is_train
        self.transform = transform
        self.count_frame = count_frame

    def __len__(self):
        return len(self.train_items)

    def __getitem__(self, idx):
        item_dir = os.path.join(self.train_dir, self.train_items[idx].id)
        frames_path = self.get_frames_path(item_dir)
        frames = []
        for frame_path in frames_path:
            frame = Image.open(frame_path).convert("RGB")
            if self.transform:
                frame = self.transform(frame)

            frames.append(torch.unsqueeze(frame, 0))

        frames = torch.cat(frames)
        frames = frames.permute(1, 0, 2, 3)
        return {"frames": frames, "label": self.train_items[idx].label}

    def get_frames_path(self, train_item_dir):
        frames = []
        for ext in [".jpg", ".JPG", ".jpeg", ".JPEG"]:
            frames.extend(glob.glob(os.path.join(train_item_dir, "*" + ext)))

        frames = sorted(frames)
        num_frames = len(frames)

        if num_frames > self.count_frame:
            frames += [frames[-1]] * (self.count_frame - num_frames)
        elif self.is_train:
            offset = numpy.random.randint(0, (num_frames - self.count_frame))
            frames = frames[offset: self.count_frame + offset]

        return frames

    def get_num_classes(self):
        return len(self.selected_labels)

    class Label:
        swiping_left = "Swiping Left"
        swiping_right = "Swiping Right"
        swiping_down = "Swiping Down"
        swiping_up = "Swiping Up"
        pushing_hand_away = "Pushing Hand Away"                         # not use
        pulling_hand_in = "Pulling Hand In"                             # not use
        sliding_two_fingers_left = "Sliding Two Fingers Left"           # not use
        sliding_two_fingers_right = "Sliding Two Fingers Right"         # not use
        sliding_two_fingers_down = "Sliding Two Fingers Down"           # not use
        sliding_two_fingers_up = "Sliding Two Fingers Up"               # not use
        pushing_two_fingers_away = "Pushing Two Fingers Away"
        pulling_two_fingers_in = "Pulling Two Fingers In"
        rolling_hand_forward = "Rolling Hand Forward"                   # not use
        rolling_hand_backward = "Rolling Hand Backward"                 # not use
        turning_hand_clockwise = "Turning Hand Clockwise"
        turning_hand_counterclockwise = "Turning Hand Counterclockwise"
        zooming_in_with_full_hand = "Zooming In With Full Hand"
        zooming_out_with_full_hand = "Zooming Out With Full Hand"
        zooming_in_with_two_fingers = "Zooming In With Two Fingers"     # not use
        zooming_out_with_two_fingers = "Zooming Out With Two Fingers"   # not use
        thumb_up = "Thumb Up"
        thumb_down = "Thumb Down"
        shaking_hand = "Shaking Hand"   # not use
        stop_sign = "Stop Sign"
        drumming_fingers = "Drumming Fingers"   # not use
        no_gesture = "No gesture"
        doing_other_things = "Doing other things"

    class Item:
        def __init__(self, id, label):
            self.id = id
            self.label = label

        # str_item = 'idx;label'
        @staticmethod
        def parse(str_item):
            str_item = str_item.split(";")
            return JesterDataset.Item(str_item[0], str_item[1])

        @staticmethod
        def get_items_from_csv(csv_path, selected_labels, max_item_per_label):
            selected_train_items = []
            with open(csv_path) as file:
                item_counter = [0 for _ in range(len(selected_labels))]
                for row in csv.reader(file):
                    item = JesterDataset.Item.parse(row[0])
                    if item.label not in selected_labels:
                        continue

                    counter_idx = selected_labels.index(item.label)
                    if item_counter[counter_idx] >= max_item_per_label:
                        continue

                    selected_train_items.append(item)
                    item_counter[counter_idx] += 1

            return selected_train_items

        @staticmethod
        def get_average_duration():
            return 16

        @staticmethod
        def get_average_size():
            return 112


if __name__ == "__main__":
    train_dir = "/home/ds/Data/academic/ta/jester/20bn-jester-v1"
    train_csv_path = "/home/ds/Data/academic/ta/jester/jester-v1-train.csv"
    selected_labels = [
        JesterDataset.Label.swiping_left,
        JesterDataset.Label.swiping_right,
    ]
    dataset = JesterDataset(
        train_dir=train_dir,
        train_csv_path=train_csv_path,
        selected_labels=selected_labels,
        max_item_per_label=2,
        is_train=True,
        transform=transforms.Compose(
            [
                transforms.CenterCrop(84),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )

    for data in dataset:
        print(data["frames"].size(), data["label"])
