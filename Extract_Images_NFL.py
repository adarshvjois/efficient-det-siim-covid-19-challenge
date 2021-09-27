#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pandarallel.pandarallel import PREFIX
import pandas as pd
import glob
import numpy
import multiprocessing
import os
import cv2

NFL_ROOT_DIR = "/home/puttuboo/Datasets/nfl-health-and-safety-helmet-assignment"
PREFIX = "train"

TRAINING_DATA = os.path.join(NFL_ROOT_DIR, f"{PREFIX}_labels.csv")

VIDEO_DIR = os.path.join(NFL_ROOT_DIR, PREFIX)
NUMBER_OF_POOLS = 4
SIZE = (1024, 1024)
OUTPUT_DIR = os.path.join(NFL_ROOT_DIR, PREFIX + "_frames")

# In[10]:


train_df = pd.read_csv(TRAINING_DATA)
labels = train_df["label"].unique()
label_per_image = train_df.groupby("video_frame")


# In[8]:


def process(file_path, prefix="train"):
    basename = os.path.basename(file_path).split(".")[0]
    print("Processing file .." + basename)
    cam = cv2.VideoCapture(file_path)

    currentframe = 0

    while True:

        # reading from frame
        ret, frame = cam.read()

        if ret:

            # if video is still left continue creating images
            name = os.path.join(OUTPUT_DIR, basename + "_" + str(currentframe) + ".jpg")
            currentframe += 1
            # writing the extracted images
            frame = cv2.resize(frame, SIZE)
            cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created

        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()


# In[9]:

if __name__ == "__main__":
    p = multiprocessing.Pool(NUMBER_OF_POOLS)
    
    try:
        # creating a folder named data
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(os.path.join(OUTPUT_DIR))
        # if not created then raise error
    except OSError:
        print("Error: Creating directory of data")
    files = [f for f in list(glob.glob(VIDEO_DIR + os.sep + "*mp4"))]
    p.map(process, files)
    p.close()
    p.join()
