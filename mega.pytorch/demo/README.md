# Demo
This repo supports two types of demo:
1. Demo on existing images/videos on the hard drive (with `demo.py`)
2. Demo on live webcam stream (with `webcam.py`)

## demo.py
Supports running on a folder of images (default) as well as on a video (`--input-video`) with all methods (base, MEGA, RDN, DFF, FGFA). It can also save results as a folder of images (default) or as a video (`--output-video`).

The default output directory is `data/output/demo` and can be customised with `--output-folder`. Filenames can also be customised with `--filename`.

### Example codes:

**Image input** (provide path to image folder with `--data-path`, and the suffix of images with `--suffix`), and **image output**. Images are saved into the folder `output-folder/filename/`.
```
python3 demo/demo.py \
    --method mega --config configs/MEGA/vid_R_101_C4_1x.yaml --checkpoint checkpoints/MEGA_R_101.pth \
    --data-path data/demo/dog --suffix ".jpg" \
    --filename dog
```

**Video input** (provide path to video file with `data-path`), and **video output**. Video is saved as `output-folder/filename.avi`.
```
python3 demo/demo.py \
    --method mega --config configs/MEGA/vid_R_101_C4_1x.yaml --checkpoint checkpoints/MEGA_R_101.pth \
    --input-video --data-path data/demo/dog.mp4 \
    --output-video --filename dog
```

## webcam.py
Supports running detection on a live webcam stream using only **MEGA or base**. Streams video recording with detection in real time, and also has the option of saving results as images (`--output-images`) or as a video (`--output-video`).

The default output directory is `data/output/webcam` and can be customised with `--output-folder`. Filenames can also be customised with `--filename`.

Press `q` at any point to end the stream.

### Example codes:

**Base** (single-frame detector). No output is saved to storage, only shown in real time.
```
python3 demo/webcam.py \
    --method base --config configs/BASE/vid_R_101_C4_1x.yaml --checkpoint checkpoints/BASE_R_101.pth
```

**MEGA**. Note the last line adjusts the offsets to enable online detection. Both a folder of images and a video will be saved at the end of the stream, with the same naming convention as in `demo.py`.

Since MEGA requires past frames to be saved to use as support frames, the setting `--max-temp-files` controls how many frames are saved before frames start to be deleted (from the front). This is especially important if detector will run for long periods at a time.
```
python3 demo/webcam.py \
    --method mega --config configs/MEGA/vid_R_101_C4_1x.yaml --checkpoint checkpoints/MEGA_R_101.pth \
    --output-video --output-images \
    --filename mega_test \
    MODEL.VID.MEGA.MIN_OFFSET -24 MODEL.VID.MEGA.MAX_OFFSET 0 MODEL.VID.MEGA.KEY_FRAME_LOCATION 24
```