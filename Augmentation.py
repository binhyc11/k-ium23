import pathlib, os, random, mplcyberpunk
import numpy as np, pandas as pd, matplotlib.pyplot as plt, tensorflow as tf
from glob import glob
from skimage.io import imread
import matplotlib.image as mpimg

from tensorflow.keras.utils import load_img, img_to_array
from keras.layers import BatchNormalization, Dense, Dropout, Flatten, MaxPool2D, Conv2D, Activation, MaxPooling2D, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator

def method (index):
    if index ==1:
        return ImageDataGenerator( horizontal_flip=True, fill_mode = "constant", cval=0)
    if index ==2:
        return ImageDataGenerator( vertical_flip=True, fill_mode = "constant", cval=0)
    if index ==3:
        return ImageDataGenerator( width_shift_range=0.1, fill_mode = "constant", cval=0)
    if index ==4:
        return ImageDataGenerator( height_shift_range=0.1, fill_mode = "constant", cval=0)
    if index ==5:
        return ImageDataGenerator( rotation_range=90, fill_mode = "constant", cval=0)
          
# def aug(path):
#         base_path = 'D:/k-ium/traindataset_2/'
#         ori_img_dir = base_path + path
#         n_imgs = len (os.listdir(ori_img_dir))
#         aug_img_dir = f'D:/aug/{path}'
#         os.makedirs(aug_img_dir, exist_ok=True)
#         list_img = os.listdir(ori_img_dir)
#         random.shuffle(list_img)
#         for img_name in list_img[:900]:
#             img = load_img(os.path.join(ori_img_dir, img_name))
#             input_arr = img_to_array(img)
#             input_arr = input_arr.reshape((1,) + input_arr.shape)
            
#             aug_method1 = random.choice([1, 2, 3, 4, 5])
#             aug_method2 = random.choice([1, 2, 3, 4, 5])
            
#             method1 = method(aug_method1)
#             method2 = method(aug_method2)
            
#             for batch in method1.flow(input_arr,
#                                      save_to_dir=aug_img_dir,
#                                      save_prefix='aug', 
#                                      save_format='jpg'):
#                 break
            
#             for batch in method2.flow(input_arr,
#                                       save_to_dir=aug_img_dir,
#                                       save_prefix='aug', 
#                                       save_format='jpg'):
#                 break
 

# for path in ['V/A/Normal/', 'V/B/Normal/']:#['I/A/Normal/', 'I/B/Normal/']:
#     aug(path)
    
         
def aug(path):
    base_path = 'D:/k-ium/traindataset_2/'+path
    for group in os.listdir(base_path):
        ori_img_dir = os.path.join(base_path, group)
        n_imgs = len (os.listdir(ori_img_dir))
        n_aug_steps = 600//n_imgs
        aug_img_dir = f'D:/aug/{path}{group}'
        os.makedirs(aug_img_dir, exist_ok=True)
        for img_name in os.listdir(ori_img_dir):
            img = load_img(os.path.join(ori_img_dir, img_name))
            input_arr = img_to_array(img)
            input_arr = input_arr.reshape((1,) + input_arr.shape)
            
            if n_aug_steps >0:
                i=0
                for batch in method(1).flow(input_arr,
                                         save_to_dir=aug_img_dir,
                                         save_prefix='hf', 
                                         save_format='jpg'):
                    i += 1
                    if i > (n_aug_steps-1):
                        break
                i=0    
                for batch in method(2).flow(input_arr,
                                         save_to_dir=aug_img_dir,
                                         save_prefix='vf', 
                                         save_format='jpg'):
                    i += 1
                    if i > (n_aug_steps-1):
                        break
                i=0    
                for batch in method(3).flow(input_arr,
                                         save_to_dir=aug_img_dir,
                                         save_prefix='ws', 
                                         save_format='jpg'):
                    i += 1
                    if i > (n_aug_steps-1):
                        break
                i=0    
                for batch in method(4).flow(input_arr,
                                         save_to_dir=aug_img_dir,
                                         save_prefix='hs', 
                                         save_format='jpg'):
                    i += 1
                    if i > (n_aug_steps-1):
                        break
                i=0    
                for batch in method(5).flow(input_arr,
                                         save_to_dir=aug_img_dir,
                                         save_prefix='r', 
                                         save_format='jpg'):
                    i += 1
                    if i > (n_aug_steps-1):
                        break

for path in ['V/B/', 'I/A/', 'I/B/', 'V/A/']:
    aug(path)
    
        