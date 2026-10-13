# UAE Speed Limit Recognition

A YOLOv8s model that finds speed limit signs and reads the number, plus an OpenCV program that runs it on a webcam and shows the current speed limit on screen.

There is no labelled dataset of UAE signs, so I trained on three public datasets from other countries (the signs use the same design: red circle, black number, normal numerals) and tested on a Saudi dataset. It recognises 20, 30, 40, 50, 60, 70, 80, 90, 100, 110 and 120 km/h.

## Results

Validation set: 2,672 images with 3,181 signs, taken from the Mapillary, TT100K and Kaggle data. This is the best epoch (73 of 83, training stopped early).

| Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|
| 0.884 | 0.689 | 0.808 | 0.666 |

| Class | P | R | mAP@50 |
|---|---|---|---|
| speed_20 | 0.905 | 0.678 | 0.787 |
| speed_30 | 0.804 | 0.624 | 0.765 |
| speed_40 | 0.876 | 0.603 | 0.800 |
| speed_50 | 0.835 | 0.605 | 0.778 |
| speed_60 | 0.867 | 0.601 | 0.769 |
| speed_70 | 0.962 | 0.701 | 0.809 |
| speed_80 | 0.855 | 0.712 | 0.833 |
| speed_90 | 0.887 | 0.582 | 0.699 |
| speed_100 | 0.927 | 0.812 | 0.913 |
| speed_110 | 0.857 | 0.815 | 0.819 |
| speed_120 | 0.944 | 0.845 | 0.920 |

Two test sets that were not used for training (output saved in `results/eval_results.txt`):

| Test set | Metric | Result |
|---|---|---|
| Kaggle test split (476 images, 513 signs) | mAP@50 | 0.980 |
| ArTS, Saudi Arabia (78 of 127 images, see below) | top-1 accuracy | 83.3% (65/78) |

The Kaggle score is high because that test split comes from the same collection as the Kaggle images I trained on, so I treat it as a check that the data pipeline works, not as a real measure. The validation score above is the one I trust more.

ArTS is a Saudi dataset with one sign per photo. Many of its signs write the number in Arabic numerals (٣٠, ٤٠, ١٠٠), which the model never saw in training, and the UAE signs I looked at on Google Street View all use normal numerals. So I went through the 127 speed limit test images and took out the 49 where the number is only written in Arabic numerals, which leaves 78. The model gets 65 of those 78 right (83.3%). By class: 30 km/h 26/26, 40 km/h 12/16, 50 km/h 15/18, 80 km/h 11/16, 60 km/h 1/2. There are no 100 km/h images left, so 60 and 100 are basically untested. I have no labelled UAE photos, so how well it works on UAE roads is not measured.

## How it works

### Data

| Source | Images used | What I did |
|---|---|---|
| [Mapillary Traffic Sign Dataset](https://www.mapillary.com/dataset/trafficsign) | 3,837 | Street photos with one JSON file per image. Converted to YOLO format, speed limits only |
| [TT100K](https://docs.ultralytics.com/datasets/detect/tt100k) | 4,845 | Only the speed limit signs (`pl*` classes). Only images that contain one are downloaded |
| [Kaggle Traffic Signs Detection](https://www.kaggle.com/datasets/pkdarabi/cardetection) | 3,292 | Speed limits only, the traffic light and stop classes are dropped. Its test folder is kept for `evaluate.py` |

That is 11,974 images: 9,302 for training and 2,672 for validation. The three datasets name their classes differently (`regulatory--maximum-speed-limit-80--g1`, `pl80`, `Speed Limit 80`) so `build_dataset.py` takes the number out of each name and maps it onto the same 11 classes. I left out 5 and 10 km/h because they aren't used in the UAE.

[ArTS](https://data.mendeley.com/datasets/4tznkn45mx/1) is only used for testing, never for training.

### Training

YOLOv8s starting from the pretrained COCO weights, 640 px images, batch size 32, up to 100 epochs, stopping if there is no improvement for 10 epochs. It stopped at epoch 83 and took about 2.6 hours on an RTX 5060 laptop GPU.

### Inference

`inference.py` runs the model on every webcam frame. It draws boxes around the signs and shows the current speed limit in a LIMIT box in the top left. The model sometimes reads a sign wrongly for a frame or two (a 60 sign read as 50, for example), so the number shown is the most common reading out of the last 15 detections. The number stays on screen for 2 seconds after the sign has gone and then disappears.

The settings are at the top of the file: `CAMERA` (I use 1 for an external webcam, 0 is usually the laptop one), `CONFIDENCE` (0.65), `SMOOTHING_FRAMES` (15) and `LIMIT_SECONDS` (2). `CAMERA` can also be the path to a video file.

## Running it

```
pip install -r requirements.txt
python inference.py
```

That uses the trained model in `models/best.pt`. Press q to quit.

To rebuild the dataset and train again, the data goes in a `data` folder in the project (it is ignored by git):

```
data/
    mapillary/       images/, annotations/ and splits/ from the Mapillary download
    tt100k.ndjson    export from the Ultralytics Platform TT100K page
    kaggle_signs/    train/, valid/, test/ and data.yaml from Kaggle
    arts/            images/ with the ArTS test images, and the test labels as a csv
```

The TT100K image links in the ndjson file only work for a while, so export a new one if the downloads fail. The ArTS labels come as an Excel file, I saved the test one as a csv with no header row.

Then run these in order:

1. `python convert_mapillary.py` makes `data/mapillary_yolo`
2. `python build_dataset.py` makes `data/speed_dataset` (downloading the TT100K images takes a while)
3. `python train.py` saves the results in `runs/detect/speed_limit_detector`
4. `python evaluate.py` tests the model in `models/best.pt` and writes `results/eval_results.txt`

The scripts stop if an output folder already exists, so delete it before running a step again. `evaluate.py` skips any ArTS csv row whose image isn't in `data/arts/images`, which is how I left out the Arabic numeral signs: I took those images out of the folder.

## Limitations

- Recall is 0.69. For 30 to 60 km/h and 90 km/h it is 0.58 to 0.62, and for 100 to 120 km/h it is 0.81 to 0.85. From the confusion matrix, about 10-17% of the signs from 20 to 90 km/h are not detected at all, and another 10-20% are read as a different speed. I think the smaller urban signs are harder, but I haven't checked that.
- Speeds next to each other get mixed up: about 10% of 50 km/h signs are read as 40 and about 7% of 80 km/h signs as 60. The smoothing hides one-off mistakes but not a sign that is read wrongly the whole time.
- It has never seen Arabic numerals.
- None of the training images are from the UAE.
- It only knows speed limits, so a different red circle sign could be read as one.
- It's a student project, don't rely on it when driving.

## What didn't work

- My first model tried to recognise 79 types of sign and only got about 0.28 mAP@50. Many classes had few images and several were red circles that look almost the same. Using just speed limits, more data from three sources and YOLOv8s brought it up to 0.81.
- I tried fine-tuning that model on TT100K afterwards. It looked better on its own validation images but was worse on the original validation set, because it had overfitted to TT100K's photos. So I merged all the sources first and trained once.

## Project files

```
models/best.pt           the trained YOLOv8s model
results/                 training curves, confusion matrices, results.csv, eval_results.txt
convert_mapillary.py     Mapillary JSON to YOLO format
build_dataset.py         merges the three datasets
train.py                 training
evaluate.py              Kaggle and ArTS tests
inference.py             webcam program
requirements.txt
```

## Licences and credits

TT100K is CC BY-NC 2.0 (non-commercial) and the model was trained on it, so the model is for non-commercial use. The Kaggle dataset is CC BY 4.0, see the Mapillary and ArTS pages for their terms. None of the datasets are included here. YOLOv8 comes from Ultralytics, which uses the AGPL-3.0 licence.

Datasets: Mapillary Traffic Sign Dataset, TT100K (Tsinghua University and Tencent), Traffic Signs Detection by P. K. Darabi (Kaggle), ArTS Arabic Traffic Sign Dataset (Mendeley Data).
