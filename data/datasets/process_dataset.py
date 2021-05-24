import os
import glob
import json
import xml.etree.ElementTree as ET

import cv2

# These 3 methods are used to convert the CalTech Pedestrian Dataset, with annotations in COCO format, to the appropriate format supported by MEGA.
# Use as reference.

base_dir = "data/datasets/custom"

def resave_images():
    if not os.path.exists(f"{base_dir}/Images"):
        os.makedirs(f"{base_dir}/Images")

    for setname in sorted(glob.glob(f'{base_dir}/original_images/*')):
        for filename in sorted(glob.glob(f'{setname}/*')):
            filename_last = filename.split("/")[-1]
            setname, videoname, frame = filename_last.split("_")
            frame = frame.split(".")[0]

            # resave images as JPEG and in separate folders
            newdir = f"{base_dir}/Images/{setname}/{videoname}"
            if not os.path.exists(newdir):
                os.makedirs(newdir)
                print(setname, videoname)

            img = cv2.imread(filename)
            cv2.imwrite(f"{newdir}/{frame}.JPEG", img)

        print(setname)

def get_imageset():
    if not os.path.exists(f"{base_dir}/ImageSets"):
        os.makedirs(f"{base_dir}/ImageSets")

    modes = ["train", "val", "test"]

    for mode in modes:
        with open(f"{base_dir}/ImageSets/{mode}.txt", "w") as f:
            f.write("")

    for setname in sorted(glob.glob(f'{base_dir}/Images/*')):
        setname = setname.split("/")[-1]

        if setname == "set09":
            mode = "val"
        elif setname == "set10":
            mode = "test"
        else:
            mode = "train"

        with open(f"{base_dir}/ImageSets/{mode}.txt", "a") as f:
            for videoname in sorted(glob.glob(f'{base_dir}/Images/{setname}/*')):
                videoname = videoname.split("/")[-1]

                files = sorted(glob.glob(f'{base_dir}/Images/{setname}/{videoname}/*'))

                for filename in files:
                    filename = filename.split("/")[-1].split(".")[0]
                    filename = int(filename)

                    f.write(f"{setname}/{videoname} 1 {filename} {len(files)}\n")

def process_json():
    with open(f"{base_dir}/all.json", "r") as f:
        all_annos = json.load(f)
    
    images = all_annos["images"]
    # list of dictionaries containing image_id, image_filename

    annotations = all_annos["annotations"]
    # list of dictionaries containing image_id, category_id (all 1s), bbox (xywh)

    # create dictionary, with every image as a key w empty list as value
    annotations_by_image = {}
    for i in range(len(images)):
        annotations_by_image[i + 1] = []

    # load data into annotations_by_image
    for annotation in annotations:
        xmin, ymin, w, h = annotation["bbox"]
        xmax = xmin + w
        ymax = ymin + h

        annotations_by_image[annotation["image_id"]].append((xmin, ymin, xmax, ymax))

    # now loop through images to save a XML file

    for image in images:
        filename = image["file_name"].split("/")[-1]
        setname, videoname, filename = filename.split("_")
        filename = filename.split(".")[0]

        vid_dir = f"{base_dir}/Annotations/{setname}/{videoname}"
        if not os.path.exists(vid_dir):
            os.makedirs(vid_dir)

        annotation = ET.Element('annotation')

        folder = ET.SubElement(annotation, 'folder')
        folder.text = f'{setname}/{videoname}'

        filename_ = ET.SubElement(annotation, 'filename')
        filename_.text = filename

        source = ET.SubElement(annotation, 'source')
        database = ET.SubElement(source, 'database')
        database.text = 'caltech'

        size = ET.SubElement(annotation, 'size')
        width = ET.SubElement(size, 'width')
        height = ET.SubElement(size, 'height')
        width.text = '640'
        height.text = '480'

        for orig_obj in annotations_by_image[image["id"]]:
            obj = ET.SubElement(annotation, 'object')
            
            name = ET.SubElement(obj, 'name')
            name.text = 'person'

            bndbox = ET.SubElement(obj, 'bndbox')
            xmax = ET.SubElement(bndbox, 'xmax')
            xmin = ET.SubElement(bndbox, 'xmin')
            ymax = ET.SubElement(bndbox, 'ymax')
            ymin = ET.SubElement(bndbox, 'ymin')

            xmax.text = str(orig_obj[2])
            xmin.text = str(orig_obj[0])
            ymax.text = str(orig_obj[3])
            ymin.text = str(orig_obj[1])

        b_xml = ET.tostring(annotation)
            
        with open(f"{vid_dir}/{filename}.xml", "wb") as f:
            f.write(b_xml)

# resave_images()
# get_imageset()
process_json()