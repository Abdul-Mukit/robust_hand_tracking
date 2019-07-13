import scipy.io as sio
import numpy as np
import os
import gc
import six.moves.urllib as urllib
import cv2
import time
import xml.etree.cElementTree as ET
import random
import shutil as sh
from shutil import copyfile
import zipfile
import shutil

from PIL import Image, ImageDraw

import csv

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def mask_hands(img, img_conv, mask):
    mask_inv = 255 - mask
    background = cv2.bitwise_and(img, img, mask=mask_inv)
    hands = cv2.bitwise_and(img_conv, img_conv, mask=mask)
    final_img = background + hands
    return final_img

def transform_hsv(img, denoise=True, random=False):
    imout = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if random==True:
        hue_values = [0, 70, 100, 120]
        hue_idx = np.random.randint(0, len(hue_values))
        factors = [hue_values[hue_idx], 0, 0]
    else:
        factors = [120, 0, 0]

    hue = imout[:, :, 0]
    sat = imout[:, :, 1]
    val = imout[:, :, 2]

    imout[:, :, 0] = np.clip(hue + factors[0], 0, 179)
    imout[:, :, 1] = np.clip(sat + factors[1], 0, 255)
    imout[:, :, 2] = np.clip(val + factors[2], 0, 255)

    imout = cv2.cvtColor(imout, cv2.COLOR_HSV2BGR)

    if denoise:
        imout = cv2.fastNlMeansDenoisingColored(imout, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

    return imout

def save_csv(csv_path, csv_content):
    with open(csv_path, 'w') as csvfile:
        wr = csv.writer(csvfile)
        for i in range(len(csv_content)):
            wr.writerow(csv_content[i])

def save_label_txt(txt_path, rows):
    out_file = open(txt_path, 'w')
    for idx_row, row in enumerate(rows):
        for idx_item, item in enumerate(row):
            out_file.write(str(item))
            if not idx_item == len(row)-1:
                out_file.write(' ')
        if not idx_row == len(rows)-1:
            out_file.write('\n')
    out_file.close()

def get_bbox_visualize(base_path, dir):
    image_path_array = []
    for root, dirs, filenames in os.walk(base_path + dir):
        for f in filenames:
            if(f.split(".")[1] == "jpg"):
                img_path = base_path + dir + "/" + f
                image_path_array.append(img_path)

    #sort image_path_array to ensure its in the low to high order expected in polygon.mat
    image_path_array.sort()
    boxes = sio.loadmat(
        base_path + dir + "/polygons.mat")
    # there are 100 of these per folder in the egohands dataset
    polygon_rows = boxes["polygons"][0]
    # first = polygons[0]
    # print(len(first))
    pointindex = 0

    for polygon_row in polygon_rows:
        font = cv2.FONT_HERSHEY_SIMPLEX

        img_id = image_path_array[pointindex]
        img = cv2.imread(img_id)

        # img_org = np.array(img)
        # img_hsv = transform_hsv(img_org, denoise=True, random=True)


        img_params = {}
        img_params["width"] = np.size(img, 1)
        img_params["height"] = np.size(img, 0)
        head, tail = os.path.split(img_id)
        img_params["filename"] = tail
        img_params["path"] = os.path.abspath(img_id)
        img_params["type"] = "train"
        pointindex += 1

        # mask = Image.new('L', (img_params["width"], img_params["height"]), 0)

        boxarray = []
        txtholder = []

        for pointlist in polygon_row:
            pst = np.empty((0, 2), int)
            max_x = max_y = min_x = min_y = height = width = 0

            findex = 0
            for point in pointlist:
                if(len(point) == 2):
                    x = int(point[0])
                    y = int(point[1])

                    if(findex == 0):
                        min_x = x
                        min_y = y
                    findex += 1
                    max_x = x if (x > max_x) else max_x
                    min_x = x if (x < min_x) else min_x
                    max_y = y if (y > max_y) else max_y
                    min_y = y if (y < min_y) else min_y
                    # print(index, "====", len(point))
                    appeno = np.array([[x, y]])
                    pst = np.append(pst, appeno, axis=0)
                    cv2.putText(img, ".", (x, y), font, 0.7,
                                (255, 255, 255), 2, cv2.LINE_AA)

            hold = {}
            hold['minx'] = min_x
            hold['miny'] = min_y
            hold['maxx'] = max_x
            hold['maxy'] = max_y

            w = np.size(img, 1)
            h = np.size(img, 0)

            if (min_x > 0 and min_y > 0 and max_x > 0 and max_y > 0):
                boxarray.append(hold)
                # labelrow = [tail, np.size(img, 1), np.size(img, 0), "hand", min_x, min_y, max_x, max_y]

                labelrow = [0, 0, 0, 0, 0]
                labelrow[1:] = convert((w, h), [min_x, max_x, min_y, max_y])
                # labelrow = [0, float(min_x)/w, float(max_x)/w, float(min_y)/h, float(max_y)/h]
                txtholder.append(labelrow)

            # if pst.any():
            #     ImageDraw.Draw(mask).polygon(pst.flatten().tolist(), outline=255, fill=255)

            cv2.polylines(img, [pst], True, (0, 255, 255), 1)
            cv2.rectangle(img, (min_x, max_y),
                          (max_x, min_y), (0, 255, 0), 1)

        # mask = np.array(mask)
        # imout = mask_hands(img_org, img_hsv, mask)

        txt_path = img_id.split(".")[0]
        if not os.path.exists(txt_path + ".txt"):
            cv2.putText(img, "DIR : " + dir + " - " + tail, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
            # cv2.imshow('Original Image ', img)
            # cv2.imshow('Masks', mask)
            # cv2.imshow('HSV Converted Image', img_hsv)
            # cv2.imshow('Final Image', imout)
            # cv2.imwrite(img_id, imout)
            
            save_label_txt(txt_path + ".txt", txtholder)
            print("===== overwriting transformed image and saving txt file for ", tail)
        cv2.waitKey(2)  # close window when a key press is detected


def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Split data, copy to train/test folders
def split_data_test_eval_train(image_dir, dest_dir):
    create_directory(dest_dir + "/" + "images")
    create_directory(dest_dir + "/" + "labels")

    data_size = 4000
    loop_index = 0
    data_sampsize = int(0.1 * data_size)
    test_samp_array = random.sample(range(data_size), k=data_sampsize)

    wd = os.getcwd()
    train_file = open(dest_dir + "/" + 'train.txt', 'w')
    test_file = open(dest_dir + "/" + 'test.txt', 'w')

    for root, dirs, filenames in os.walk(image_dir):
        for dir in dirs:
            for f in os.listdir(image_dir + dir):
                if(f.split(".")[1] == "jpg"):
                    loop_index += 1
                    print(loop_index, f)

                    shutil.move(image_dir + dir + "/" + f, dest_dir + "/" + "images/" + f)
                    shutil.move(image_dir + dir + "/" + f.split(".")[0] + ".txt",
                              dest_dir + "/" + "labels/" + f.split(".")[0] + ".txt")

                    if loop_index in test_samp_array:
                        test_file.write('%s/images/%s\n'%(dest_dir, f))
                    else:
                        train_file.write('%s/images/%s\n'%(dest_dir, f))

                    print(loop_index, image_dir + f)
            print(">   done scanning director ", dir)
            os.remove(image_dir + dir + "/polygons.mat")
            os.rmdir(image_dir + dir)

        print("Train/test content generation complete!")
        test_file.close()
        train_file.close()
        # generate_label_files("images/")


def generate_csv_files(image_dir):
    for root, dirs, filenames in os.walk(image_dir):
        for dir in dirs:
            get_bbox_visualize(image_dir, dir)

def generate_txt_files(image_dir):
    for root, dirs, filenames in os.walk(image_dir):
        for dir in dirs:
            get_bbox_visualize(image_dir, dir)

    print("Txt generation complete!\nGenerating train/test list-txt files")
    split_data_test_eval_train(image_dir="egohands/_LABELLED_SAMPLES/",
                               dest_dir=dest_dir)



# rename image files so we can have them all in a train/test/eval folder.
def rename_files(image_dir):
    print("Renaming files")
    loop_index = 0
    for root, dirs, filenames in os.walk(image_dir):
        for dir in dirs:
            for f in os.listdir(image_dir + dir):
                if (dir not in f):
                    if(f.split(".")[1] == "jpg"):
                        loop_index += 1
                        os.rename(image_dir + dir +
                                  "/" + f, image_dir + dir +
                                  "/" + dir + "_" + f)
                else:
                    break

        generate_txt_files("egohands/_LABELLED_SAMPLES/")

def extract_folder(dataset_path):
    print("Egohands dataset already downloaded.\nGenerating CSV files")
    # if not os.path.exists("egohands"):
    if True:
        zip_ref = zipfile.ZipFile(dataset_path, 'r')
        print("> Extracting Dataset files")
        zip_ref.extractall("egohands")
        print("> Extraction complete")
        zip_ref.close()
        rename_files("egohands/_LABELLED_SAMPLES/")

def download_egohands_dataset(dataset_url, dataset_path):
    is_downloaded = os.path.exists(dataset_path)
    if not is_downloaded:
        print(
            "> downloading egohands dataset. This may take a while (1.3GB, say 3-5mins). Coffee break?")
        opener = urllib.request.URLopener()
        opener.retrieve(dataset_url, dataset_path)
        print("> download complete")
        extract_folder(dataset_path);

    else:
        extract_folder(dataset_path)


EGOHANDS_DATASET_URL = "http://vision.soic.indiana.edu/egohands_files/egohands_data.zip"
EGO_HANDS_FILE = "egohands_data.zip"
dest_dir="/media/drive1/Datasets/egohands_test"


download_egohands_dataset(EGOHANDS_DATASET_URL, EGO_HANDS_FILE)
