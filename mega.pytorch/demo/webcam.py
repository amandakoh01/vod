import os
import glob
import cv2
import argparse
import random

from predictor import VIDDemo
from mega_core.config import cfg

parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam")
parser.add_argument(
    "--method",
    choices=["base", "dff", "fgfa", "rdn", "mega"],
    default="base",
    type=str,
    help="which method to use",
)
parser.add_argument(
    "--config",
    default="configs/BASE/vid_R_101_C4_1x.yaml",
    help="path to config file",
)
parser.add_argument(
    "--checkpoint",
    default="checkpoints/BASE_R_101.pth",
    help="The path to the checkpoint for test.",
)
parser.add_argument(
    "--output-images",
    action="store_true",
    help="if True, output images.",
)
parser.add_argument(
    "--output-video",
    action="store_true",
    help="if True, output a video.",
)
parser.add_argument(
    "--output-folder",
    default="data/output/webcam",
    help="Path to save temporary imgs and visualisation files"
)
parser.add_argument(
    "--filename",
    default="webcam",
    help="what to name the video file or the image folder result"
)
parser.add_argument(
    "opts",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)

args = parser.parse_args()

cfg.merge_from_file("configs/BASE_RCNN_1gpu.yaml")
cfg.merge_from_file(args.config)
cfg.merge_from_list(["MODEL.WEIGHT", args.checkpoint])
cfg.merge_from_list(args.opts)

output_folder = args.output_folder

# make output/method folder
if not os.path.exists(os.path.join(output_folder, args.method)):
    os.makedirs(os.path.join(output_folder, args.method))

# make temp folder / remove temp data
if os.path.exists(os.path.join(output_folder, "temp")):
    files = glob.glob(output_folder + "/temp/*")
    for f in files:
        os.remove(f)
else:
    os.makedirs(os.path.join(output_folder, "temp"))

# set up video stream
video = cv2.VideoCapture(0)
vid_demo = VIDDemo(
    cfg,
    method=args.method,
    confidence_threshold=0.7,
    output_folder=output_folder,
    output_name=args.filename
)

idx = 0

visualization_results = []

while True:
    ret, img_orig = video.read()
    img_cur = vid_demo.perform_transform(img_orig)

    if args.method == "base":
        visualization_result = vid_demo.run_on_image(img_orig, img_cur)
    
    elif args.method == "mega":
        cv2.imwrite(os.path.join(output_folder, "temp", "%06d.JPEG" % idx), img_orig)

        infos = {}
        infos["cur"] = img_cur
        infos["frame_category"] = 0 if idx == 0 else 1
        infos["seg_len"] = idx + 1
        infos["pattern"] = output_folder + "/temp/%06d"
        infos["img_dir"] = "%s" + ".JPEG"
        infos["transforms"] = vid_demo.build_pil_transform()

        # first image in seq
        if idx == 0:
            infos["ref_g"] = []
            for i in range(cfg.MODEL.VID.MEGA.GLOBAL.SIZE):
                infos["ref_g"].append(img_cur)

            infos["ref_l"] = [img_cur]

        # rest of images
        else:
            g_idx = random.choice(range(idx))
            g_filename = infos["pattern"] % g_idx
            g_img = cv2.imread(infos["img_dir"] % g_filename)
            g_img = vid_demo.perform_transform(g_img)
            infos["ref_g"] = [g_img]

            l_idx = idx - 1
            l_filename = infos["pattern"] % l_idx
            l_img = cv2.imread(infos["img_dir"] % l_filename)
            l_img = vid_demo.perform_transform(l_img)
            infos["ref_l"] = [l_img]

        visualization_result = vid_demo.run_on_image(img_orig, infos)

    else:
        raise ValueError("Only base and mega supported")

    visualization_results.append(visualization_result)
    cv2.imshow('video', visualization_result)

    idx += 1

    if cv2.waitKey(1) == ord('q'):
        break

if args.output_images:
    vid_demo.generate_images(visualization_results)
if args.output_video:
    vid_demo.generate_video(visualization_results)