"""
Builds a single speed limit dataset from three sources:
  1. Mapillary - data/mapillary_yolo (made by convert_mapillary.py)
  2. TT100K    - data/tt100k.ndjson (the images get downloaded from it)
  3. Kaggle    - data/kaggle_signs (already in YOLO format)

Each source names its classes differently, so everything is converted to the same 11 classes: speed_20, speed_30 ... speed_120.

Writes to:  data/speed_dataset/   (images/, labels/, data.yaml)
"""

import json
import re
import shutil
import sys
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent

MAPILLARY_DIR = ROOT / "data" / "mapillary_yolo"
TT100K_FILE = ROOT / "data" / "tt100k.ndjson"
KAGGLE_DIR = ROOT / "data" / "kaggle_signs"
OUTPUT_DIR = ROOT / "data" / "speed_dataset"

# 5 and 10 km/h are left out because they aren't used in the UAE
SPEEDS = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

class_names = []
speed_to_class = {}

for i in range(len(SPEEDS)):
    class_names.append(f"speed_{SPEEDS[i]}") # convert them to speed_XX format for YOLO
    speed_to_class[SPEEDS[i]] = i


# use regex to extract speed from the class name (e.g. "speed_limit_30" = 30)
def get_speed(name):
    match = re.search(r"\d+", name)
    if match is None:
        return None
    return int(match.group())


def remap_labels(label_path, class_map):
    # reads a YOLO label file and swaps the old class number to a new shared class number
    new_lines = []

    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 0:
            continue

        old_class = int(parts[0])
        if old_class in class_map:
            new_lines.append(str(class_map[old_class]) + " " + " ".join(parts[1:]))

    return new_lines


def find_image(folder, name):
    for ext in [".jpg", ".jpeg", ".png"]:
        path = folder / (name + ext)
        if path.exists():
            return path
    return None


# check everything is where it should be before doing any work
if not MAPILLARY_DIR.exists():
    sys.exit(f"Can't find {MAPILLARY_DIR} - run convert_mapillary.py first")
if not TT100K_FILE.exists():
    sys.exit(f"Can't find {TT100K_FILE} - see the README")
if not KAGGLE_DIR.exists():
    sys.exit(f"Can't find {KAGGLE_DIR} - see the README")
if OUTPUT_DIR.exists():
    sys.exit(f"{OUTPUT_DIR} already exists - delete it first so old files don't get mixed in")

for split in ["train", "val"]:
    (OUTPUT_DIR / "images" / split).mkdir(parents=True)
    (OUTPUT_DIR / "labels" / split).mkdir(parents=True)

mapillary_count = 0
tt100k_count = 0
kaggle_count = 0

# ---------- Mapillary ----------

print("Mapillary...")

with open(MAPILLARY_DIR / "data.yaml") as f:
    mapillary_names = yaml.safe_load(f)["names"]

mapillary_map = {}

for i in range(len(mapillary_names)):
    speed = get_speed(mapillary_names[i])
    if speed in speed_to_class:
        mapillary_map[i] = speed_to_class[speed]

for split in ["train", "val"]:
    image_folder = MAPILLARY_DIR / "images" / split
    label_folder = MAPILLARY_DIR / "labels" / split

    for label_path in label_folder.glob("*.txt"):

        new_lines = remap_labels(label_path, mapillary_map)
        image_path = image_folder / (label_path.stem + ".jpg")
        if len(new_lines) == 0 or not image_path.exists():
            continue

        shutil.copy(image_path, OUTPUT_DIR / "images" / split / f"mapillary_{label_path.stem}.jpg")
        (OUTPUT_DIR / "labels" / split / f"mapillary_{label_path.stem}.txt").write_text("\n".join(new_lines))
        mapillary_count += 1

print(f"  {mapillary_count} images")

# ---------- TT100K ----------
# The ndjson has one line per image. The first line has the class names, the other lines have the image url and the boxes as [class, x, y, w, h]

print("TT100K...")

with open(TT100K_FILE, encoding="utf-8") as f:
    ndjson_lines = f.read().splitlines()

header = json.loads(ndjson_lines[0])
tt100k_map = {}

for index_text in header["class_names"]:
    name = header["class_names"][index_text]
    if name.startswith("pl"):  # pl = prohibitory speed limit (pm and il are other speed signs)
        speed = get_speed(name)
        if speed in speed_to_class:
            tt100k_map[int(index_text)] = speed_to_class[speed]

# TT100K train and val both go into the train set, its test split becomes the val set
tt100k_split = {"train": "train", "val": "train", "test": "val"}

# find images that have a speed limit sign, only those get downloaded
to_download = []

for line in ndjson_lines[1:]:
    record = json.loads(line)
    if record.get("type") != "image":
        continue

    boxes = record.get("annotations", {}).get("boxes", [])
    new_lines = []

    for box in boxes:
        if box[0] in tt100k_map:
            new_lines.append(f"{tt100k_map[box[0]]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}")
    if len(new_lines) > 0:
        to_download.append([record, new_lines])

print(f"  {len(to_download)} images have speed limit signs, downloading (this takes a while)...")

failed = 0

for number in range(len(to_download)):
    record = to_download[number][0]
    new_lines = to_download[number][1]
    split = tt100k_split.get(record.get("split"), "train")
    stem = Path(record["file"]).stem

    try:
        response = requests.get(record["url"], timeout=30)
        response.raise_for_status()

    except Exception as error:
        failed += 1
        print(f"  failed: {record['file']} ({error})")
        continue

    (OUTPUT_DIR / "images" / split / f"tt100k_{stem}.jpg").write_bytes(response.content)
    (OUTPUT_DIR / "labels" / split / f"tt100k_{stem}.txt").write_text("\n".join(new_lines))
    tt100k_count += 1

    if (number + 1) % 100 == 0:
        print(f"  {number + 1} / {len(to_download)}")

print(f"  {tt100k_count} images ({failed} failed)")

# ---------- Kaggle ----------
print("Kaggle...")

with open(KAGGLE_DIR / "data.yaml") as f:
    kaggle_names = yaml.safe_load(f)["names"]

# the traffic light and stop classes in this dataset are ignored
kaggle_map = {}
for i in range(len(kaggle_names)):
    if "speed limit" in kaggle_names[i].lower():
        speed = get_speed(kaggle_names[i])
        if speed in speed_to_class:
            kaggle_map[i] = speed_to_class[speed]

# Kaggle's test folder is not used here, it's kept for evaluate.py
kaggle_split = {"train": "train", "valid": "val"}

for folder_name in kaggle_split:
    split = kaggle_split[folder_name]
    image_folder = KAGGLE_DIR / folder_name / "images"
    label_folder = KAGGLE_DIR / folder_name / "labels"

    for label_path in label_folder.glob("*.txt"):
        new_lines = remap_labels(label_path, kaggle_map)
        image_path = find_image(image_folder, label_path.stem)

        if len(new_lines) == 0 or image_path is None:
            continue

        shutil.copy(image_path, OUTPUT_DIR / "images" / split / f"kaggle_{label_path.stem}{image_path.suffix}")
        (OUTPUT_DIR / "labels" / split / f"kaggle_{label_path.stem}.txt").write_text("\n".join(new_lines))
        kaggle_count += 1

print(f"  {kaggle_count} images")

# ---------- data.yaml ----------
yaml_text = f"train: {(OUTPUT_DIR / 'images' / 'train').as_posix()}\n"
yaml_text += f"val: {(OUTPUT_DIR / 'images' / 'val').as_posix()}\n"
yaml_text += f"nc: {len(class_names)}\n"
yaml_text += f"names: {class_names}\n"

(OUTPUT_DIR / "data.yaml").write_text(yaml_text)

total = mapillary_count + tt100k_count + kaggle_count
print(f"\nDone: {mapillary_count} Mapillary + {tt100k_count} TT100K + {kaggle_count} Kaggle = {total} images")
print(f"Wrote {OUTPUT_DIR / 'data.yaml'}")
