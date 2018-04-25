import os
import os.path as osp
import sys

import cv2
import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import gluon
from os.path import exists
from os import mkdir
from time import time
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, '..'))

def mk_if(path):
    if not exists(path):
        mkdir(path)

from dataloader.transform import transform_histgram



class histgram():
    def __init__(self,croprecord_filename):
        self.crop_record = pd.read_csv(croprecord_filename,index_col='imgname')

    def hist_and_norm(self, imgs):
        res = []
        for i in range(len(imgs)):
            img = imgs[i]
            img = cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
            m = np.mean(np.transpose(img,(2,0,1)),axis=(1,2))
            res.append(m)
        return np.mean(res,axis=0)
        #histbase = np.mean(img,axis=1)
        #return histbase

    def get_histgram(self, rec_path,indexs):
        dataset = gluon.data.vision.ImageRecordDataset(filename=rec_path)
        data = gluon.data.dataloader.DataLoader(dataset, batch_size=1, shuffle=False)
        res = []
        tic = time()
        for idx, (img, l) in enumerate(data):
            img = img[0]
            imgname = indexs[idx]
            if imgname in self.crop_record.index:
                dfi = self.crop_record.loc[imgname]
                df = dfi[dfi.score >= 0.25]
                if len(df) == 0:
                    df = dfi.head(1)
                df = df[df.score >= 0.24]
                height,width,_ = img.shape
                data = []
                for i in df.itertuples():
                    xmin = int(i.xmin * width)
                    xmax = int(i.xmax * width)
                    ymin = int(i.ymin * height)
                    ymax = int(i.ymax * height)
                    data.append(img[ymin:ymax,xmin:xmax].asnumpy())
                if len(df) == 0:
                    data = [mx.image.CenterCropAug((112, 112))(img).asnumpy(),]
            else:
                data = [mx.image.CenterCropAug((112, 112))(img).asnumpy(),]
            if idx % 100 == 0:
                print('idx %s ,time used: %s'%(idx,time()-tic))
            res_temp =  self.hist_and_norm(data)
            #print(res_temp.shape)
            res.append(res_temp)
        return np.stack(res)

    def compare_histgram(self, h1, h2):
        return cv2.compareHist(h1, h2, 0)

    def generate_histgram_database(self, rec_path, npy_filepath,index_path):
        with open(index_path,'r') as f:
            indexs = list(map(lambda x:x.strip(),f.readlines()))
        all_hists = self.get_histgram(rec_path,indexs)
        mk_if(osp.dirname(npy_filepath))
        np.save( npy_filepath,all_hists)


if __name__ == '__main__':
    script_path = osp.dirname(osp.realpath(__file__))
    model = histgram(osp.join(script_path,'../../clothdata/ssd_res_maincolor.csv'))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", help="which database to predict")
    parser.add_argument("--index_path", help="rec to imgname")
    parser.add_argument("--output_dir", help="which dir to output")
    args = parser.parse_args()
    for i in args.database.split(','):
        model.generate_histgram_database(osp.join(script_path,'../data/{}.rec'.format(i))\
                                         ,osp.join(script_path,'../data/hist',args.output_dir\
                                                   ,'{}_hist.npy'.format(osp.basename(i))),
                                         osp.join(script_path,
                                                  '../data/CLOTHdevkit/clothall/ImageSets'
                                                  ,args.index_path))
