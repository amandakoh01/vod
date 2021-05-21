# Introduction
Forked from [official MEGA repository](https://github.com/Scalsol/mega.pytorch)

**Additions:**
1. Docker container
2. Online webcam demo
3. Built-in custom dataset implementation

# Docker commands

## Docker build
The folder structure is split into 4 main parts to layerise the Docker container:
1. **dependencies** -- copied and installed first
2. **mega_build** (contains mega_core and setup.py) -- copied first and setup.py is run
3. **mega.pytorch** (contains all other folders and is the work directory) -- copied after mega_core is built
4. **data** - the data folder in mega.pytorch is empty. Use -v to link with an external data folder. The folder should contain the following subdirectories:
 - datasets (for train/test)
 - demo (videos or images to predict/demo on)
 - output (where output images/videos will be stored)

If you need to rebuild the docker container yourself, note that GPU access is needed when building. Ensure
```
RUN python3 -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)'
```
returns True.
Follow [steps here](https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime).

## Docker run
To use webcam demo, run 
```
xhost + 
```

then
```
sudo docker run -it --gpus all --ipc=host \
-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/video0 \
-v /home/dh/Desktop/vod/data:/workspace/mega.pytorch/data \
mega 
```

## Run train, test, demo and webcam

To run with multiple GPUs add this in front of the python3 call.

```
python3 -m torch.distributed.launch \
    --nproc_per_node 4 \
```

### Train
```
python3 tools/train_net.py \
--config-file configs/MEGA/vid_R_101_C4_1x.yaml \
OUTPUT_DIR output/training/test \
DATASETS.TRAIN VID_custom_train DATASETS.TEST VID_custom_test
```

### Test
```
python3 tools/test_net.py 
--config-file configs/MEGA/vid_R_101_C4_1x.yaml \
MODEL.WEIGHT checkpoints/MEGA_OWN_R_101.pth \
OUTPUT_DIR output/testing/MEGA_R_101_1x \
DATSETS.TRAIN VID_custom_train DATASETS.TEST VID_custom_test
```

### Demo
Demo supports running on a folder of images (default) as well as a video (using the flag `--input-video`). Provide path to the folder containing images, or the video respectively.

It also has the option of returning results as a folder of images (default, saves to the folder `output-folder/filename`) or as a video file (using the flag `--output-video`, saves to the folder `output-folder` with filename `filename.avi`).

```
python3 demo/demo.py \
--method mega --config configs/MEGA/vid_R_101_C4_1x.yaml --checkpoint checkpoints/MEGA_R_101.pth \
--input-video --data-path data/demo/dog.mp4 \
--output-video --filename dog
```

```
python3 demo/demo.py \
--method mega --config configs/MEGA/vid_R_101_C4_1x.yaml --checkpoint checkpoints/MEGA_R_101.pth \
--data-path data/demo/dog --suffix ".jpg" \
--filename dog

```

### Webcam
Supports running detection on a live webcam stream. Streams video recording with detection in real time, and also has the option of saving results as images (`--output-images`) or as a video (`--output-video`).

Note the last line adjusts the offsets for MEGA to enable online detection.

```
python3 demo/webcam.py \
--method mega --config configs/MEGA/vid_R_101_C4_1x.yaml --checkpoint checkpoints/MEGA_R_101.pth \
--output-video --output-images \
MODEL.VID.MEGA.MIN_OFFSET -24 MODEL.VID.MEGA.MAX_OFFSET 0 MODEL.VID.MEGA.KEY_FRAME_LOCATION 24
```

# Custom datasets
There are generic custom TRAIN, VAL and TEST datasets pre-registered. To use your own custom dataset, create a folder `data/datasets/custom` (do not change the word "custom")

The folder should contain the following folders:
- **Annotations**
- **Images**
- **ImageSets** (with text files train.txt, val.txt and test.txt)

Look at the sample files given as an example of how they should be structured. Frames from a single video should be all in one folder, with file names running from 000000.JPEG to xxxxxx.JPEG (6 digits).

Once the data is in the desired format, use the datasets by giving the config settings
```
DATSETS.TRAIN VID_custom_train DATASETS.TEST VID_custom_test
```

For more details on implementing your own custom dataset, go to [CUSTOMIZE.md on the original repo](https://github.com/Scalsol/mega.pytorch/blob/master/CUSTOMIZE.md)

Notes:
- If encountering a list index error when using datasets (train or test), delete files in the data/datasets/cache folder
- The VOD frameworks require 4 strings in each line of the ImageSets text: `folder_directory` `no meaning (can just put 1)` `frame number (0 indexed)` `video length / number of frames`