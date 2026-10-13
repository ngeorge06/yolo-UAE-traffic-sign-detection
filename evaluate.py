"""
Tests the trained model on two test sets it wasn't trained on.

Kaggle test split: has real bounding boxes, so Ultralytics gives the normal
detection scores (precision, recall, mAP).

ArTS test images: no bounding boxes, just one label per image. For each image
I take the most confident detection and check if it is the right speed.

"""

import csv
import re
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best.pt"
KAGGLE_DIR = ROOT / "data" / "kaggle_signs"
KAGGLE_EVAL_DIR = ROOT / "data" / "kaggle_test_eval"  # made by this script
ARTS_IMAGES = ROOT / "data" / "arts" / "images"
ARTS_LABELS = ROOT / "data" / "arts" / "Labels_Test_Arabic_Traffic_Signs_24_518.csv"
RESULTS_FILE = ROOT / "results" / "eval_results.txt"

# ArTS label number -> speed (the ArTS filenames look like "speed_30-1")
ARTS_SPEEDS = {16: 100, 17: 30, 18: 40, 19: 50, 20: 60, 21: 80}

output_lines = []


def say(text=""):
    print(text)
    output_lines.append(text)


def get_speed(name):
    # first number in the name, "speed_80" -> 80, "Speed Limit 80" -> 80
    match = re.search(r"\d+", name)
    return int(match.group())


def find_image(folder, name):
    for ext in [".jpg", ".jpeg", ".png"]:
        path = folder / (name + ext)
        if path.exists():
            return path
    return None


def most_common(items):
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    best = None
    for item in counts:
        if best is None or counts[item] > counts[best]:
            best = item
    return best, counts[best]


def test_kaggle(model):
    # the model's class number for each speed
    speed_to_class = {}
    for class_id in model.names:
        speed_to_class[get_speed(model.names[class_id])] = class_id

    # Kaggle class number -> model class number (speed limits only)
    with open(KAGGLE_DIR / "data.yaml") as f:
        kaggle_names = yaml.safe_load(f)["names"]
    class_map = {}
    for i in range(len(kaggle_names)):
        if "speed limit" in kaggle_names[i].lower():
            speed = get_speed(kaggle_names[i])
            if speed in speed_to_class:
                class_map[i] = speed_to_class[speed]

    # copy the test images that have speed signs into a new folder with the labels
    # renumbered to match the model
    (KAGGLE_EVAL_DIR / "images").mkdir(parents=True, exist_ok=True)
    (KAGGLE_EVAL_DIR / "labels").mkdir(parents=True, exist_ok=True)
    image_count = 0
    for label_path in (KAGGLE_DIR / "test" / "labels").glob("*.txt"):
        new_lines = []
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) > 0 and int(parts[0]) in class_map:
                new_lines.append(str(class_map[int(parts[0])]) + " " + " ".join(parts[1:]))
        image_path = find_image(KAGGLE_DIR / "test" / "images", label_path.stem)
        if len(new_lines) == 0 or image_path is None:
            continue
        shutil.copy(image_path, KAGGLE_EVAL_DIR / "images" / image_path.name)
        (KAGGLE_EVAL_DIR / "labels" / label_path.name).write_text("\n".join(new_lines))
        image_count += 1

    names_list = []
    for class_id in range(len(model.names)):
        names_list.append(model.names[class_id])
    images_path = (KAGGLE_EVAL_DIR / "images").as_posix()
    yaml_text = f"train: {images_path}\n"  # ultralytics needs a train line even though it isn't used
    yaml_text += f"val: {images_path}\n"
    yaml_text += f"nc: {len(names_list)}\n"
    yaml_text += f"names: {names_list}\n"
    (KAGGLE_EVAL_DIR / "data.yaml").write_text(yaml_text)

    box = model.val(data=str(KAGGLE_EVAL_DIR / "data.yaml"), plots=False, verbose=False).box

    say(f"Kaggle test split ({image_count} images with speed limit signs)")
    say("class            P      R   mAP50  mAP50-95")
    for i in range(len(box.ap_class_index)):
        name = model.names[int(box.ap_class_index[i])]
        say(f"{name:<11}{box.p[i]:>7.3f}{box.r[i]:>7.3f}{box.ap50[i]:>8.3f}{box.ap[i]:>10.3f}")
    name = "all"
    say(f"{name:<11}{box.mp:>7.3f}{box.mr:>7.3f}{box.map50:>8.3f}{box.map:>10.3f}")
    say()


def test_arts(model):
    seen = {}
    correct = {}
    wrong_answers = {}
    for speed in ARTS_SPEEDS.values():
        seen[speed] = 0
        correct[speed] = 0
        wrong_answers[speed] = []
    no_image = 0

    with open(ARTS_LABELS, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) != 2 or not row[1].strip().isdigit():
                continue
            label = int(row[1])
            if label not in ARTS_SPEEDS:
                continue  # not a speed limit sign
            speed = ARTS_SPEEDS[label]

            image_path = find_image(ARTS_IMAGES, row[0].strip())
            if image_path is None:
                no_image += 1  # these are the Arabic numeral images I took out
                continue

            # take the most confident detection in the image
            result = model.predict(str(image_path), conf=0.25, verbose=False)
            answer = "no detection"
            best_conf = 0
            for box in result[0].boxes:
                if float(box.conf[0]) > best_conf:
                    best_conf = float(box.conf[0])
                    answer = model.names[int(box.cls[0])]

            seen[speed] += 1
            if answer == f"speed_{speed}":
                correct[speed] += 1
            else:
                wrong_answers[speed].append(answer)

    total_seen = 0
    total_correct = 0
    for speed in seen:
        total_seen += seen[speed]
        total_correct += correct[speed]

    say(f"ArTS test images (top-1 accuracy, {total_seen} images)")
    say("class       accuracy  correct   most common wrong answer")
    for speed in sorted(seen):
        if seen[speed] == 0:
            continue  # no images left for this speed
        accuracy = f"{100 * correct[speed] / seen[speed]:.1f}%"
        count = f"{correct[speed]}/{seen[speed]}"
        wrong_text = "-"
        if len(wrong_answers[speed]) > 0:
            answer, times = most_common(wrong_answers[speed])
            wrong_text = f"{answer} (x{times})"
        name = "speed_" + str(speed)
        say(f"{name:<11}{accuracy:>9}{count:>9}   {wrong_text}")
    overall = f"{100 * total_correct / total_seen:.1f}%"
    overall_count = f"{total_correct}/{total_seen}"
    name = "overall"
    say(f"{name:<11}{overall:>9}{overall_count:>9}")
    say(f"({no_image} CSV rows had no image, those are the removed Arabic numeral signs)")
    say()


def main():
    model = YOLO(str(MODEL_PATH))
    say("Model: " + MODEL_PATH.name)
    say()

    test_kaggle(model)
    test_arts(model)

    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"Saved to {RESULTS_FILE}")


# needed on Windows because ultralytics starts worker processes
if __name__ == "__main__":
    main()
