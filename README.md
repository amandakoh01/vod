# Introduction
Forked from [official MEGA repository](https://github.com/Scalsol/mega.pytorch)

**Additions:**
1. Docker container
2. Online webcam demo
3. Built-in custom dataset implementation

# Using the repo
The folder structure is split into 4 main parts to layerise the Docker container:
1. **dependencies** -- copied and installed first
2. **mega_build** (contains mega_core and setup.py) -- copied first and setup.py is run
3. **mega.pytorch** (contains all other folders and is the work directory) -- copied after mega_core is built
4. **data** - the data folder in mega.pytorch is empty. Use -v to link with an external data folder. The folder should contain the following subdirectories:
  1. datasets (for train/test)
  2. demo (videos or images to predict/demo on)
  3. output (where output images/videos will be stored)

Several pre-trained models are provided in the Docker container in `mega.pytorch/checkpoints`, but you can use any models even outside of the container.

For details on building your own Docker container from the source files, scroll down to the **Rebuild docker image** section.

### Docker run
This line is only needed to run the webcam demo.
```
xhost + 
```

Run the docker container:
```
sudo docker run -it --gpus all --ipc=host \
-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/video0 \
-v /home/dh/Desktop/vod/data:/workspace/mega.pytorch/data \
mega 
```

### Multiple GPUs

To run with multiple GPUs, add this in front of the python3 call.

```
python3 -m torch.distributed.launch \
    --nproc_per_node 2 \
```

### Train and test
There are a number of datasets supported by MEGA. The default from the original repository is to use ILSVRC VID images.

In order to modify which datasets are used, you can modify `data/datasets/datasets.yaml` (see `mega_build/mega_core/config/paths_catalog` for dataset names), and then use the `--dataset-config-file` flag with an additional config file as shown below.

Also see **Custom datasets** section below to customise your own datasets.


```
python3 tools/train_net.py \
--config-file configs/MEGA/vid_R_101_C4_1x.yaml \
--dataset-config-file data/datasets/datasets.yaml \
OUTPUT_DIR data/output/training/training1
```

```
python3 tools/test_net.py \
--config-file configs/MEGA/vid_R_101_C4_1x.yaml \
--dataset-config-file data/datasets/datasets.yaml \
MODEL.WEIGHT checkpoints/MEGA_R_101.pth \
OUTPUT_DIR data/output/testing/testing1
```

### Demo
Demo supports running on a folder of images (default) as well as a video (`--input-video`). It can also save results as a folder of images (default) or as a video (`--output-video`).

Default output directory is `data/output/demo` and can be customised with `--output-folder`.

Video input (provide path to video file), and video output (saves `output-folder/filename.avi`)
```
python3 demo/demo.py \
--method mega --config configs/MEGA/vid_R_101_C4_1x.yaml --checkpoint checkpoints/MEGA_R_101.pth \
--input-video --data-path data/demo/dog.mp4 \
--output-video --filename dog
```

Image input (provide path to image folder, and the suffix of images), and image output (saves images into `output-folder/filename/`).
```
python3 demo/demo.py \
--method mega --config configs/MEGA/vid_R_101_C4_1x.yaml --checkpoint checkpoints/MEGA_R_101.pth \
--data-path data/demo/dog --suffix ".jpg" \
--filename dog
```

### Webcam
Supports running detection on a live webcam stream using only **MEGA or base**. Streams video recording with detection in real time, and also has the option of saving results as images (`--output-images`) or as a video (`--output-video`).

```
python3 demo/webcam.py \
--method base --config configs/BASE/vid_R_101_C4_1x.yaml --checkpoint checkpoints/BASE_R_101.pth \
--output-video --output-images \
--filename base_test
```

Note the last line adjusts the offsets for MEGA to enable online detection. There is an additional setting `--max-temp-files` to control how many temp files to save to disk that global frames can be drawn from (if detector will run for long periods at a time).
```
python3 demo/webcam.py \
--method mega --config configs/MEGA/vid_R_101_C4_1x.yaml --checkpoint checkpoints/MEGA_R_101.pth \
--output-video --output-images \
--filename mega_test \
MODEL.VID.MEGA.MIN_OFFSET -24 MODEL.VID.MEGA.MAX_OFFSET 0 MODEL.VID.MEGA.KEY_FRAME_LOCATION 24
```

# Custom datasets
There are generic custom TRAIN, VAL and TEST datasets pre-registered. To use your own custom dataset, create a folder `data/datasets/custom` (do not change the word "custom")

The folder should contain the following folders:
- **Annotations**
- **Images**
- **ImageSets** (with text files train.txt, val.txt and test.txt)

Look at the sample files given as an example of how they should be structured. Frames from a single video should be all in one folder, with file names running from 000000.JPEG to xxxxxx.JPEG (6 digits).

For more details on implementing your own custom dataset, go to [CUSTOMIZE.md on the original repo](https://github.com/Scalsol/mega.pytorch/blob/master/CUSTOMIZE.md)

Notes:
- If encountering a list index error when using datasets (train or test), delete files in the data/datasets/cache folder
- The VOD frameworks require 4 strings in each line of the ImageSets text: `folder_directory` `no meaning (can just put 1)` `frame number (0 indexed)` `video length / number of frames`

# Rebuild docker image

1. Enable GPU access is needed when building. Follow [steps here](https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime). Ensure line 85 in the dockerfile returns True when building.
2. Download [apex](https://github.com/NVIDIA/apex) and [cocoapi](https://github.com/cocodataset/cocoapi) into `dependencies/apex` and `dependencies/cocoapi` (same level as mega_build, mega.pytorch)
3. Download any pre-trained checkpoints into `mega.pytorch/checkpoints` if you want. The [official repository](https://github.com/Scalsol/mega.pytorch) contains download links.
