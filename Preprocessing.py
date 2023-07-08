import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os, utils, cv2, numpy as np, pandas as pd
from PIL import Image
from skimage import img_as_float
from skimage import exposure

class Filterting ():
    def __init__(self, image, kernel_size, threshold):
        self.image = image
        self.size = kernel_size
        self.kernel = np.ones(kernel_size)
        self.threshold = threshold

    def divide_grid (self):
        n_row, n_col = self.image.shape
        
        if n_row%self.size == 0:
            n_grid_r = n_row / self.size
        else:
            n_grid_r = n_row // self.size + 1
            
        if n_col%self.size == 0:
            n_grid_c = n_col / self.size
        else:
            n_grid_c = n_col // self.size +1
            
        padding = np.zeros((int(n_grid_r)*self.size, int(n_grid_c)*self.size)).astype('uint8')
        padding[:n_row, :n_col] = self.image
        
        return int(n_grid_r),int(n_grid_c), padding
    
    def get_grid (self, row, col):
        grid = [(self.size*row), (self.size*(row+1)), (self.size*col), (self.size*(col+1))]
        return grid    
        
    def cal_ratio (self, grid):
        summation = np.multiply (grid, self.kernel).sum()/255
        ratio = summation/ (self.size*self.size)
        return ratio
    
    def conv (self):
        n_grid_r,n_grid_c, img_padded = self.divide_grid()
        
        RC = []
        for r in range(n_grid_r):
            for c in range(n_grid_c):
                grid = self.get_grid (r, c)
                ratio = self.cal_ratio (img_padded[grid[0]:grid[1], grid[2]:grid[3]])
                if ratio < self.threshold:
                    RC.append([r,c])
                    
        temp_pad = np.ones(img_padded.shape)
        for rc in RC:
            zero_grid = self.get_grid (rc[0], rc[1])
            temp_pad[zero_grid[0]:zero_grid[1], zero_grid[2]:zero_grid[3]] = np.zeros(self.size)
        img_padded = np.multiply (temp_pad, img_padded)
        return img_padded
    
def remove_upper (array, bound):
    a = np.where(array>bound, bound, array)
    a = torange(a)
    return a

def torange(array):
    minimum = array.min()
    maximum = array.max()
    new_array = (array-minimum)/(maximum-minimum)*255
    return new_array.astype('uint8')

def crop (array):
    r, c = array.shape
    if r==c:
        return array[10:r-10,10:c-10]
    else:
        diff = int((c-r)/2)
        img = array[10:(r-10),(10+diff):(c-10-diff)]
        img2 = remove_ouliers(img)
        return img2

def removeBG(array):
    BG = array[:50,:].mean()
    return remove_upper(array, BG)

def remove_ouliers(array):
    ave = array.mean()
    img = np.where(array>245, ave, array)
    return img

def processing(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = crop(gray)
    p2, p98 = np.percentile(gray, (2, 98))
    img_rescale = exposure.rescale_intensity(gray, in_range=(p2, p98))
    img = removeBG(img_rescale)

    ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [cnt],-1, 255, -1)
    
    kernel = np.ones((5,5),np.uint8)
    dilate = cv2.dilate(mask,kernel,iterations=2)
    new_mask = np.where(dilate>0, 1, 0)
    return gray.astype('uint8'), new_mask.astype('uint8')

data_dir = os.listdir(r'D:\k-ium\train_raw\train_set')
for i in range (len(data_dir)):
    path = os.path.join(r'D:\k-ium\train_raw\train_set', data_dir[i])
    new_img, mask = processing(path)
    masked_img = np.multiply(new_img, mask)
    im = Image.fromarray(masked_img, 'L')
    im.save(f'D:/k-ium/preprocessing/{data_dir[i]}')
