from matplotlib import pyplot
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
import os
import re
import cv2
import numpy

def implot(im):
    pyplot.figure(figsize=(8, 8))
    pyplot.gray()
    pyplot.axis('off')
    pyplot.imshow(im)
    return pyplot.show()

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def find_desired_size_for_padding(data_dir='data/all/'):
    sh0, sh1 = 0, 0
    for sub_dir in sorted(os.listdir(data_dir), key=natural_keys):
        if not sub_dir.startswith('.'):
            files = os.listdir(os.path.join(data_dir, sub_dir))
            dsm = None
            dtm = None
            rgb = None
            edge = None
            for f in files:
                filename, file_extension = os.path.splitext(f)
                if 'DSM' in filename:
                    dsm = cv2.imread(os.path.join(data_dir, sub_dir, f), cv2.IMREAD_LOAD_GDAL)
                elif 'DTM' in filename:
                    dtm = cv2.imread(os.path.join(data_dir, sub_dir, f), cv2.IMREAD_LOAD_GDAL)
                elif 'RGB' in filename and file_extension == '.tif':
                    rgb = cv2.imread(os.path.join(data_dir, sub_dir, f), cv2.IMREAD_COLOR)
                elif 'EDGE' in filename:
                    edge = cv2.imread(os.path.join(data_dir, sub_dir, f), cv2.IMREAD_GRAYSCALE)
            tmp_shape = rgb.shape[:2]
            if tmp_shape[0] > sh0:
                sh0 = tmp_shape[0]
            if tmp_shape[1] > sh1:
                sh1 = tmp_shape[1]
    return max([sh0, sh1])

def im_resize(im, size=(256, 256)):
    new_im = cv2.resize(im, size)
    return new_im

def pad_im(im, size):
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(size) / max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = size - new_size[1]
    delta_h = size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    return new_im

def transform_data(data_dir='data/all/', desired_size=262, save_data=False):
    X = []
    y = []
    if save_data is True:
        os.system('mkdir -p data/padded/')
    for sub_dir in sorted(os.listdir(data_dir), key=natural_keys):
        if not sub_dir.startswith('.'):
            files = os.listdir(os.path.join(data_dir, sub_dir))
            dsm = None
            dtm = None
            rgb = None
            gray = None
            edge = None
            for f in files:
                filename, file_extension = os.path.splitext(f)
                if 'DSM' in filename:
                    dsm = cv2.imread(os.path.join(data_dir, sub_dir, f), cv2.IMREAD_LOAD_GDAL)
                elif 'DTM' in filename:
                    dtm = cv2.imread(os.path.join(data_dir, sub_dir, f), cv2.IMREAD_LOAD_GDAL)
                elif 'RGB' in filename and file_extension == '.tif':
                    rgb = cv2.imread(os.path.join(data_dir, sub_dir, f), cv2.IMREAD_COLOR)
                    gray = cv2.imread(os.path.join(data_dir, sub_dir, f), cv2.IMREAD_GRAYSCALE)
                elif 'EDGE' in filename:
                    edge = cv2.imread(os.path.join(data_dir, sub_dir, f), cv2.IMREAD_GRAYSCALE)
            elevation = numpy.nan_to_num(dsm - dtm)
            new_rgb = pad_im(rgb, size=desired_size)
            new_gray = numpy.expand_dims(pad_im(gray, size=desired_size), axis=2)
            new_elevation = numpy.expand_dims(pad_im(elevation, size=desired_size), axis=2)
            rgbe = numpy.concatenate((new_rgb, new_elevation), axis=2)
            new_edge = numpy.expand_dims(pad_im(edge, size=desired_size), axis=3)
            new_edge[new_edge > 0] = 1
            # X.append(numpy.expand_dims(im_resize(new_gray), axis=0))
            X.append(numpy.expand_dims(numpy.expand_dims(im_resize(new_gray), axis=0), axis=3))
            y.append(numpy.expand_dims(numpy.expand_dims(im_resize(new_edge), axis=0), axis=3))
    return numpy.concatenate(X, axis=0), numpy.concatenate(y, axis=0)
