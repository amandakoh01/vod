# Video Object Detection
Forked from [official MEGA repository](https://github.com/Scalsol/mega.pytorch).

**Additions:**
1. Docker container
2. Online webcam demo
3. Simplified built-in custom dataset implementation

## Using the repo
The folder structure is split into 4 main parts to layerise the Docker container:
1. **dependencies** -- copied and installed first
2. **mega_build** (contains mega_core and setup.py) -- copied first and setup.py is run
3. **mega.pytorch** (contains all other folders and is the work directory) -- copied after mega_core is built
4. **data** - the data folder in mega.pytorch is empty. Use -v to link with an external data folder. The folder must contain a `datasets` folder to do training/testing, and it is also suggested to have `demo` and `output` folders to organise data.

Notes:
- To provide flexibility with customising classes to train on or detect, classes (for all purposes - train, test, demo, webcam) are taken from `mega_build/mega_core/class_labels.py`. In order to customise the classes to a specific dataset, use -v to link with an external `class_labels.py` file.
- Several pre-trained models are pre-provided in the Docker container in `mega.pytorch/checkpoints`, but technically any models even outside of the container can be used (just mount into the workspace with -v).

## Docker run
This line is only needed to run the webcam demo.
```
xhost + 
```

Run the docker container (second line only needed to run the webcam demo):
```
sudo docker run -it --gpus all --ipc=host \
-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/video0 \
-v /home/dh/Desktop/vod/data:/workspace/mega.pytorch/data \
-v /home/dh/Desktop/vod/class_labels.py:/workspace/mega.pytorch/mega_core/class_labels.py \
mega 
```

After opening the Docker container, follow these links to find out more about the individual Python commands:
- [Train and test](https://github.com/amandakoh01/vod/tree/main/mega.pytorch/tools)
- [Demo (images/videos on disk or webcam stream)](https://github.com/amandakoh01/vod/tree/main/mega.pytorch/demo)

## Rebuild docker image

1. Download [apex](https://github.com/NVIDIA/apex) and [cocoapi](https://github.com/cocodataset/cocoapi) into `dependencies/apex` and `dependencies/cocoapi` (same level as mega_build, mega.pytorch)
2. Download desired pre-trained checkpoints to ship with the Docker container into `mega.pytorch/checkpoints`. The [official repository](https://github.com/Scalsol/mega.pytorch) contains download links to models trained on ILSVRC2015 Videos.
3. GPU access is needed when building. Follow [steps here](https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime) to enable. Ensure line 85 in the dockerfile returns True when building.