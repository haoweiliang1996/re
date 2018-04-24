from os.path import join
import os

import cv2
import numpy as np
import pandas as pd

from .imdb import Imdb


class Fashion(Imdb):
    def __init__(self, label_path, datatype='train'):
        super().__init__('fashion')
        assert datatype in ['train', 'test', 'val']
        self.label_path = label_path
        with open(join(label_path, 'Eval/list_eval_partition.txt'), 'r') as fp:
            with open(join(label_path, 'Anno/list_bbox.txt'), 'r') as fb:
                with open(join(label_path, 'Anno/list_landmarks.txt'), 'r') as fl:
                    _ = int(fp.readline().strip())
                    _ = int(fb.readline().strip())
                    _ = int(fl.readline().strip())
                    partition = pd.read_csv(fp, delimiter=' *')
                    bbox = pd.read_csv(fb, delimiter=' *')
                    landmarks = pd.read_csv(fl, delimiter=' *')
                    bbox = pd.concat((bbox, landmarks['clothes_type'], partition['evaluation_status']), axis=1)
                    self.bbox = bbox.groupby('evaluation_status').get_group(datatype)
        self.num_images = len(self.bbox)

    def label_from_index(self, index):
        img_info = self.bbox.iloc[index]
        image_file = self.image_path_from_index(index)
        assert os.path.isfile(image_file), 'Path does not exist: {}'.format(image_file)
        y,x = cv2.imread(image_file).shape[:2]
        xmin = float(img_info.x_1) / x
        ymin = float(img_info.y_1) / y
        xmax = float(img_info.x_2) / x
        ymax = float(img_info.y_2) / y
        return np.array([[img_info.clothes_type, xmin, ymin, xmax, ymax],])

    def image_path_from_index(self, index):
        return join(self.label_path, 'Img',self.bbox.iloc[index].image_name)


if __name__ == '__main__':
    imdb = Fashion('../data/FashionLandmarkDetectionBenchmark')
    imdb.label_from_index(1)
