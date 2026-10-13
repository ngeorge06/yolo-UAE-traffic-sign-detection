"""
Converts the Mapillary Traffic Sign Dataset annotations (one JSON file per
image) into YOLO format. Only the speed limit signs are kept.

Reads from:  data/mapillary/   (images/, annotations/, splits/)
Writes to:   data/mapillary_yolo/   (images/, labels/, data.yaml)
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAPILLARY_DIR = ROOT / "data" / "mapillary"
OUTPUT_DIR = ROOT / "data" / "mapillary_yolo"

# Speed limit classes to keep. The 5 and 10 km/h ones are dropped later in build_dataset.py because those signs aren't used in the UAE.
SPEED_CLASSES = [
    "regulatory--maximum-speed-limit-5--g1",
    "regulatory--maximum-speed-limit-10--g1",
    "regulatory--maximum-speed-limit-20--g1",
    "regulatory--maximum-speed-limit-30--g1",
    "regulatory--maximum-speed-limit-40--g1",
    "regulatory--maximum-speed-limit-50--g1",
    "regulatory--maximum-speed-limit-60--g1",
    "regulatory--maximum-speed-limit-70--g1",
    "regulatory--maximum-speed-limit-80--g1",
    "regulatory--maximum-speed-limit-90--g1",
    "regulatory--maximum-speed-limit-100--g1",
    "regulatory--maximum-speed-limit-120--g1",
]

if not MAPILLARY_DIR.exists():
    sys.exit(f"Can't find {MAPILLARY_DIR} - see the README for the data folder layout")
if OUTPUT_DIR.exists():
    sys.exit(f"{OUTPUT_DIR} already exists - delete it first so old labels don't get mixed in")

# class name converted to a class number
class_ids = {}
sign_counts = {}
for i in range(len(SPEED_CLASSES)):
    class_ids[SPEED_CLASSES[i]] = i
    sign_counts[SPEED_CLASSES[i]] = 0

# the test split has no public labels so only train and val are used
for split in ["train", "val"]:
    (OUTPUT_DIR / "images" / split).mkdir(parents=True)
    (OUTPUT_DIR / "labels" / split).mkdir(parents=True)

    keys = (MAPILLARY_DIR / "splits" / f"{split}.txt").read_text().splitlines()
    image_count = 0

    for key in keys:
        image_path = MAPILLARY_DIR / "images" / f"{key}.jpg"
        annotation_path = MAPILLARY_DIR / "annotations" / f"{key}.json"
        if not image_path.exists() or not annotation_path.exists():
            continue

        with open(annotation_path, encoding="utf-8") as f:
            annotation = json.load(f)
        width = annotation["width"]
        height = annotation["height"]

        label_lines = []
        for sign in annotation["objects"]:
            label = sign["label"]
            box = sign["bbox"]

            if label not in class_ids:
                continue
            if "cross_boundary" in box:
                continue  # sign wraps around the edge of a panorama, skip it

            # YOLO wants the box centre and size as fractions of the image size
            x_centre = (box["xmin"] + box["xmax"]) / 2 / width
            y_centre = (box["ymin"] + box["ymax"]) / 2 / height
            box_width = (box["xmax"] - box["xmin"]) / width
            box_height = (box["ymax"] - box["ymin"]) / height

            label_lines.append(f"{class_ids[label]} {x_centre:.6f} {y_centre:.6f} {box_width:.6f} {box_height:.6f}")
            sign_counts[label] += 1

        if len(label_lines) == 0:
            continue  # no speed limit signs in this image

        shutil.copy(image_path, OUTPUT_DIR / "images" / split / f"{key}.jpg")
        (OUTPUT_DIR / "labels" / split / f"{key}.txt").write_text("\n".join(label_lines))
        image_count += 1

    print(f"{split}: {image_count} images")

# data.yaml tells YOLO where the images are and what the classes are called
yaml_text = f"train: {(OUTPUT_DIR / 'images' / 'train').as_posix()}\n"
yaml_text += f"val: {(OUTPUT_DIR / 'images' / 'val').as_posix()}\n"
yaml_text += f"nc: {len(SPEED_CLASSES)}\n"
yaml_text += f"names: {SPEED_CLASSES}\n"
(OUTPUT_DIR / "data.yaml").write_text(yaml_text)

print("\nsigns per class:")
for name in SPEED_CLASSES:
    print(f"  {name}: {sign_counts[name]}")
print(f"\nDone, wrote {OUTPUT_DIR / 'data.yaml'}")
