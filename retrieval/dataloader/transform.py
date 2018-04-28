# transform for all process
import mxnet as mx
import numpy as np
import cv2

from mxnet import image
from mxnet import nd

augs = image.CreateAugmenter(data_shape=(3, 224, 224), rand_mirror=True, rand_crop=True, rand_resize=False,
                             brightness=0.125, contrast=0.125, saturation=0.125)
augs_val = image.CreateAugmenter(data_shape=(3, 224, 224), rand_crop=False, rand_mirror=False)
cropaug = mx.image.RandomSizedCropAug(size=(224, 224), min_area=0.5, ratio=[0.75, 1.33333])
use_rgb = True
use_color = True
resize_short_aug = mx.image.ResizeAug(size=224)
augs.insert(0, resize_short_aug)
augs_val.insert(0, resize_short_aug)


def transform(data, label):
    for aug in augs:
        data = aug(data)
    return nd.transpose(mx.image.color_normalize(data.astype(np.float32) / 255,
                                                 mean=mx.nd.array([0.485, 0.456, 0.406]),
                                                 std=mx.nd.array([0.229, 0.224, 0.225])), (2, 0, 1)), label


def transform_val(data, label):
    for aug in augs_val:
        data = aug(data)
    return nd.transpose(mx.image.color_normalize(data.astype(np.float32) / 255,
                                                 mean=mx.nd.array([0.485, 0.456, 0.406]),
                                                 std=mx.nd.array([0.229, 0.224, 0.225])), (2, 0, 1)), label

def fix_crop(src, tu):
    x, y = tu
    return mx.image.fixed_crop(src, x, y, 224, 224)
def trans(img,label,crop):
    img = mx.image.resize_short(img, 224)
    img = crop(img)
    return nd.transpose(mx.image.color_normalize(img.astype(np.float32) / 255,
                                                    mean=mx.nd.array([0.485, 0.456, 0.406]),
                                                    std=mx.nd.array([0.229, 0.224, 0.225])), (2, 0, 1)),label

def transform_test(img):
    def fix_crop(src, tu):
        x, y = tu
        return mx.image.fixed_crop(src, x, y, 224, 224)

    img = mx.image.resize_short(img, 224)
    height, width, _ = img.shape
    imgs = [fix_crop(img, tu) for tu in [(0, 0), (width - 224, 0), (0, height - 224), (width - 224, height - 224)]]
    imgs.append(mx.image.center_crop(img, (224, 224))[0])
    temp = []
    for i in imgs:
        temp.append(nd.transpose(mx.image.color_normalize(i.astype(np.float32) / 255,
                                                     mean=mx.nd.array([0.485, 0.456, 0.406]),
                                                     std=mx.nd.array([0.229, 0.224, 0.225])), (2, 0, 1)))
    return nd.stack(*temp)

def transform_pad(data,label):
    img = data.asnumpy()
    height, width, _ = img.shape
    fill_value = [0,0,0]
    if height <= width:
        short_height = True
    else:
        short_height = False
    long_side = max(height,width)
    ratio = 224 / long_side
    img = cv2.resize(img,(0,0),fx = ratio, fy = ratio)
    height, width, _ = img.shape
    short_side = min(height,width)
    pad_width = int((224 - short_side) / 2)
    if short_height:
        img = cv2.copyMakeBorder(img,pad_width,pad_width,0,0,cv2.BORDER_CONSTANT,value=fill_value)
    else:
        img = cv2.copyMakeBorder(img,0,0,pad_width,pad_width,cv2.BORDER_CONSTANT,value=fill_value)
    img = cv2.resize(img,(224,224))
    img = mx.nd.array(img)
    return nd.transpose(mx.image.color_normalize(img.astype(np.float32) / 255,
                                                    mean=mx.nd.array([0.485, 0.456, 0.406]),
                                                    std=mx.nd.array([0.229, 0.224, 0.225])), (2, 0, 1)),label

def transform_padlong(data,label):
    resize_short_width = 180
    img = image.resize_short(data, resize_short_width)
    fill_value = [0,0,0]
    pad_width = 120
    img = img.asnumpy()
    img = cv2.copyMakeBorder(img,pad_width,pad_width,pad_width,pad_width,cv2.BORDER_CONSTANT,value=fill_value)
    img = mx.nd.array(img)
    img, _ = image.center_crop(img, (180, 320))
    return nd.transpose(mx.image.color_normalize(img.astype(np.float32) / 255,
                                                    mean=mx.nd.array([0.485, 0.456, 0.406]),
                                                    std=mx.nd.array([0.229, 0.224, 0.225])), (2, 0, 1)),label

patch_width = 56
patch_pad = int(patch_width / 2)
def fix_crop_hist(src, tu):
    x, y = tu
    return mx.image.fixed_crop(src, x, y, patch_width, patch_width)
def crop1_hist(img):
    height, width, _ = img.shape
    height, width, _ = img.shape
    return fix_crop_hist(img,(patch_pad,patch_pad))

def crop2_hist(img):
    height, width, _ = img.shape
    height, width, _ = img.shape
    return fix_crop_hist(img,(width - patch_width - patch_pad, patch_pad))

def crop4_hist(img):
    height, width, _ = img.shape
    height, width, _ = img.shape
    return fix_crop_hist(img,(patch_pad, height - patch_width - patch_pad))

def crop3_hist(img):
    height, width, _ = img.shape
    height, width, _ = img.shape
    return fix_crop_hist(img,(width - patch_width - patch_pad, height - patch_width - patch_pad))

def crop5_hist(img):
    height, width, _ = img.shape
    height, width, _ = img.shape
    return mx.image.center_crop(img, (patch_width, patch_width))[0]

def transform_histgram(data, label):
    # return data,label
    data = mx.image.CenterCropAug((112, 112))(data)

    return data, label

def transform_histgram_around(data, label):
    # return data,label
    data = mx.nd.stack(crop1_hist(data),crop2_hist(data),crop3_hist(data),crop4_hist(data),crop5_hist(data))
    #data = [eval('crop%s_hist(data)'%i) for i in range(1,6)]
    return data, label

if __name__ == '__main__':
    for i in augs:
        print(i.dumps())
