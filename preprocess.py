import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
import os
import csv
import shutil
import math
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

    arg_parser.add_argument('--output-folder', type=str, default='dataset', help='''
        Preprocessed dataset output folder
    ''')

    return arg_parser.parse_args()

excluded = [
    "Portrait of the Postman Joseph Roulin 7.jpg",
    "Portrait of the Postman Joseph Roulin 8.jpg",
    "Portrait of Doctor Felix Rey 2.jpg",
    "Anna Cornelia van Gogh.jpg",
    "Anna Cornelia van Gogh 3.jpg"
    "Johanna Bonger, Vincent van Goghs sister-in-law.jpg",
    "Johanna van Gogh-Bonger, Vincent van Goghs sister-in-law.jpg",
    "The painter Emile Bernard and Vincent van Gogh seen from the back have a good time discussing color insinde a black-and-wite photo.jpg",
    "The Ravoux Inn Auvers-sur-Oise, France.jpg",
    "Theo van Gogh, the brother of Vincent 2.jpg",
    "Theo van Gogh, the brother of Vincent 3.jpg",
    "Theo van Gogh, the brother of Vincent 4.jpg",
    "Theo van Gogh, the brother of Vincent.jpg",
    "Theodorus van Gogh.jpg",
    "Vincent van Gogh, age 13.jpg",
    "Vincent van Gogh, age 19.jpg",
    "L Arlesienne Madame Ginoux 5.jpg",
    "Portrait of Doctor Gachet L Homme a la Pipe.jpg",
    "Vincents brother Theo died one year after him. They were buried side by side as equals. I do not know, why Theos wife Johanna, when her time came, was nor buried next to them.jpg"
    "Digger.jpg",
    "Farmer Leaning on his Spade.jpg"
]

dims = (150,150)


def load_image(path):
    return cv2.imread(path)

def store_image(path, img):
    cv2.imwrite(path, img)

def to_saturation(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv[:,:,1]

def filter_img(img):
    sat_img = to_saturation(img)
    sat = np.mean(sat_img)
    if sat < 60:
        return False
    return True

def check_bar(img):
    kernel = np.ones((5,1,3), dtype=np.uint8)*255
    print(kernel)
    print(kernel.shape)
    print(img.shape)
    return cv2.matchTemplate(255-img,kernel,cv2.TM_CCOEFF)
    
def clean_string(string):
    return string.replace(" ", "_").lower()

def x_shear(img, cols, rows, shear_percent):
    M = np.float32([[1, shear_percent, 0], [0, 1, 0], [0, 0, 1]])
    shear_margin = int(math.ceil(shear_percent * rows))
    sheared_img = cv2.warpPerspective(img,M,(int(cols + 4*shear_margin),int(rows)))[:,shear_margin:(cols - 2*shear_margin),:]
    return sheared_img

def y_shear(img, cols, rows, shear_percent):
    M = np.float32([[1, 0, 0], [shear_percent, 1, 0], [0, 0, 1]])
    shear_margin = int(math.ceil(shear_percent * cols))
    sheared_img = cv2.warpPerspective(img,M,(int(cols),int(rows + 4*shear_margin)))[shear_margin:(rows - 2*shear_margin),:,:]
    return sheared_img

def sheared_images(img):
    rows, cols, dim = img.shape
    yield img
    for xshr in [0, 0.04, 0.08]:
        yield x_shear(img, cols, rows, xshr)
    for yshr in [0, 0.04, 0.08]:
        yield y_shear(img, cols, rows, yshr)
    
def brightness_adj_images(img):
    for bright_adjust in [0.9, 1, 1.1]:
        hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        hsv_img[:,:,2] = np.clip(hsv_img[:,:,2]*bright_adjust, 0, 255)
        yield cv2.cvtColor(hsv_img,cv2.COLOR_HSV2BGR)

def square_patches(img):
    rows, cols, dim = img.shape
    if rows > cols:
        rectangleness = ((rows - cols) / cols) // .1
        steps = int(math.ceil(rectangleness / 2))
        stepsize = int(math.floor((rows - cols)/(2*max(1,steps - 1))))
        for step in range(0, steps):
            yield img[stepsize*step:stepsize*step+cols,:,:]
        for step in range(0, steps):
            yield img[rows-stepsize*step-cols:rows-stepsize*step,:,:]
    else:
        rectangleness = ((cols - rows) / rows) // .1
        steps = int(math.ceil(rectangleness / 2))
        stepsize = int(math.floor((cols - rows)/(2*max(1,steps - 1))))
        for step in range(0, steps):
            yield img[:,stepsize*step:stepsize*step+rows,:]
        for step in range(0, steps):
            yield img[:,cols-stepsize*step-rows:cols-stepsize*step,:]
    return img

def noised_images(img):
    yield img
    #yield np.clip(img + np.random.normal(0, 2, img.shape).astype(int), 0, 255).astype(np.uint8)
    yield np.clip(img + np.random.normal(0, 5, img.shape).astype(int), 0, 255).astype(np.uint8)

def augment_image(img):
    for b_img in brightness_adj_images(img):
        for adj_img in sheared_images(b_img):
            for sqr_img in square_patches(adj_img):
                resized = cv2.resize(sqr_img, dims, interpolation = cv2.INTER_AREA )
                for noise_img in noised_images(resized):
                    yield noise_img

if __name__ == '__main__':
    args = get_args()

    os.makedirs(args.output_folder, exist_ok=True)

    i = 0
    for theme in os.listdir(args.dataset_folder):
        theme_dir = os.path.join(args.dataset_folder, theme)
        if os.path.isdir(theme_dir) and theme not in ["Drawings", "Sketches in letters"]:
            print(theme)
            fmt_theme = clean_string(theme)
            for img_name in os.listdir(theme_dir):
                if img_name not in excluded:
                    print(img_name)
                    img = load_image( os.path.join(theme_dir, img_name) )
                    if filter_img(img):
                        for aug_img in augment_image(img):
                            out_path = os.path.join(args.output_folder, "%05d.jpg" % i  )
                            store_image(out_path, aug_img)
                            i += 1
                            if i > 20000:
                                exit()