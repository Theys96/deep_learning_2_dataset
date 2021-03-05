import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
import os
import csv
import shutil
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from skimage import filters
import scipy
from argparse import ArgumentParser
if os.path.dirname(__file__) != '':
    os.chdir(os.path.dirname(__file__))

def get_args():
    arg_parser = ArgumentParser()
    
    arg_parser.add_argument('--dataset-folder', type=str, default='van-gogh-paintings', help='''
        Dataset folder
    ''')

    arg_parser.add_argument('--output-folder', type=str, default='saturation_pictures', help='''
        Preprocessed dataset output folder
    ''')

    arg_parser.add_argument('--filtered-output-folder', type=str, default='saturation_pictures_filtered', help='''
        Filtered images output folder
    ''')

    arg_parser.add_argument('--saturation-threshold', type=int, default=65, help='''
        Minimum saturation value to filter at 
    ''')

    return arg_parser.parse_args()



def load_image(path):
    return cv2.imread(path)

def store_image(path, img):
    cv2.imwrite(path, img)

def to_saturation(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv[:,:,1]

def clean_string(string):
    return string.replace(" ", "_").lower()


if __name__ == '__main__':
    args = get_args()

    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(args.filtered_output_folder, exist_ok=True)

    avg_saturations = list()
    for theme in os.listdir(args.dataset_folder):
        theme_dir = os.path.join(args.dataset_folder, theme)
        if os.path.isdir(theme_dir) and theme not in ["Drawings", "Sketches in letters"]:
            print(theme)
            fmt_theme = clean_string(theme)
            for img_name in os.listdir(theme_dir):
                print("\t" + img_name)
                img = load_image( os.path.join(theme_dir, img_name) )
                sat_img = to_saturation(img)
                sat = np.mean(sat_img)
                avg_saturations.append(sat)
                img_out_name = ("%03d_%03d.jpg" % (int(sat), 1000*(sat - int(sat)))) + f"__{fmt_theme}__{clean_string(img_name)}"
                if sat > args.saturation_threshold:
                    store_image( os.path.join(args.output_folder, img_out_name), img)
                else:
                    store_image( os.path.join(args.filtered_output_folder, img_out_name), img)

    
    bins = np.linspace(0., 255., 50)
    fig = plt.figure()
    ax = plt.gca()
    ax.set_title("Distribution of samples by saturation")
    ax.set_ylabel("Number of samples")
    ax.set_xlabel("Saturation")
    ax.hist(avg_saturations, bins, label='Saturation levels')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("saturation_experiment.pdf")
    plt.savefig("saturation_experiment.png")