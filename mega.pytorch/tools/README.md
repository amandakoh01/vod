# Train and test

## Preparing the datasets
There are a number of datasets supported by MEGA. The default from the original repository is to use ILSVRC VID images. See `mega_build/mega_core/config/paths_catalog` for dataset names and details (e.g. where to store the data).

To simplify the use of custom datasets, generic custom TRAIN, VAL and TEST datasets have been pre-registered. To use a custom dataset, create a folder `data/datasets/custom` (do not change the word "custom").

The folder should contain the following folders:
- **Images**: Frames from a single video should be all in one folder, with file names running from 000000.JPEG to xxxxxx.JPEG (all 6 digits).
- **ImageSets** (with text files train.txt, val.txt and test.txt): Each text file should contain 4 strings in each line: `directory to the video folder` `no meaning (can just put 1)` `frame number (0 indexed)` `video length / number of frames`
- **Annotations**: One XML file is needed per image, regardless of whether there are objects in the frame. Annotations should have the exact same file structure as Images. Follow the PASCAL VOC format, with required information: size (height, width) and object (name, bndbox)).

**Look at the [sample files given](https://github.com/amandakoh01/vod/tree/main/data/datasets/custom) for examples.** The folder structure should look like this:

```
data
├── datasets
|   ├── custom
|   |   |── Images
|   |   |   |── video_1
|   |   |   |   |── 000000.JPEG
|   |   |   |   |── 000001.JPEG
|   |   |   |   |── 000002.JPEG
|   |   |   |   ...
|   |   |   |── video_2
|   |   |   |   |── 000000.JPEG
|   |   |   |   |── 000001.JPEG
|   |   |   |   |── 000002.JPEG
|   |   |   |   ...
|   |   |   ...
|   |   |── Annotations
|   |   |   |── video_1
|   |   |   |   |── 000000.XML
|   |   |   |   |── 000001.XML
|   |   |   |   |── 000002.XML
|   |   |   |   ...
|   |   |   |── video_2
|   |   |   |   |── 000000.XML
|   |   |   |   |── 000001.XML
|   |   |   |   |── 000002.XML
|   |   |   |   ...
|   |   |   ...
|   |   |── ImageSets
|   |   |   |── train.txt
|   |   |   |── val.txt
```

Notes:
- Datasets are cached in `data/datasets/cache`. Delete this folder if encountering errors or if the dataset is changed.
- Remember to edit `class_labels.py` when using a custom dataset.

## Running

In order to modify which datasets are used, modify `data/datasets/datasets.yaml` (with `VID_custom_train`, `VID_custom_val` or `VID_custom_test`, as well as specifying the number of classes + background). For instance:

``` yaml
DATASETS:
  TRAIN: ("VID_custom_train",)
  TEST: ("VID_custom_val",)
MODEL:
  ROI_BOX_HEAD:
    NUM_CLASSES: 31 # (this should be the number of classes + 1 to account for background)
```

Then, when running the train/test script, use the `--dataset-config-file` flag to specify this dataset config file.

**Train**
```
python3 tools/train_net.py \
    --config-file configs/MEGA/vid_R_101_C4_1x.yaml \
    --dataset-config-file data/datasets/datasets.yaml \
    OUTPUT_DIR data/output/training/training1 \
    SOLVER.MAX_ITER 100000
```

**Test**
```
python3 tools/test_net.py \
    --config-file configs/MEGA/vid_R_101_C4_1x.yaml \
    --dataset-config-file data/datasets/datasets.yaml \
    MODEL.WEIGHT checkpoints/MEGA_R_101.pth \
    OUTPUT_DIR data/output/testing/testing1
```

## Multiple GPUs
To run with multiple GPUs, add this in front of the python3 call.

```
python3 -m torch.distributed.launch \
    --nproc_per_node 2 \
```

For instance:
```
python3 -m torch.distributed.launch \
    --nproc_per_node 2 \
    tools/train_net.py \
    --config-file configs/MEGA/vid_R_101_C4_1x.yaml \
    --dataset-config-file data/datasets/datasets.yaml \
    OUTPUT_DIR data/output/training/training1
```
