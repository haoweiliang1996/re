'''
    允许自定义输入的文件库，支持保存本地索引库，argmax
    do PCA,and l2_norm
'''

import os
import sys

from logger import logger

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, '../../'))
sys.path.append(os.path.join(curr_path, '..'))
import os.path as osp

# define a simple data batch

import mxnet as mx
import numpy as np
import cv2
from mxnet import nd
from time import time

from retrieval.dataloader.transform import transform_test
from retrieval.mxsymbol import symbol_factory


class Retrieval_model():
    def __init__(self, ctx, first_class_id):
        self.DEBUG = False
        self.prefix = 'retrieval'
        self.first_class_id = str(first_class_id)
        # self.pca = decomposition.PCA(n_components=128)
        '''
        with open(osp.join(self.prefix,'checkpoint','PCA_model.pickle'),'rb') as f:
            self.pca = pickle.load(f)
        '''
        self.ctx = ctx
        self.model = self.get_mod(folder_name='retrieval/checkpoint/%d' % self.first_class_id,
                                  checkpoint_name='net_best.params', ctx=ctx)
        # self.anchors_data, self.cropus_data = self.load_search_database('database_sm128')
        self.database = self.load_search_database(
            [osp.join('database', self.first_class_id), ])

        (self.cropus_index, self.cropus_index_inverse) = self.load_search_index()
        self.cropus_hist = self.load_hist_database()

    def load_hist_database(self):
        base = osp.join('retrieval/cropus/hist', self.first_class_id)
        hist_path = osp.join(base, 'cropus_hist.npy')
        assert osp.exists(hist_path), 'hist %s not found' % hist_path
        return np.load(hist_path)

    def get_hist(self, img):
        if not isinstance(img, nd.NDArray):
            img1 = nd.array(img)
        else:
            img1 = img
        if int(self.first_class_id) != 4:
            img1 = mx.img.resize_short(img1, 224)
            img1 = mx.img.center_crop(img1, (112, 112))[0].asnumpy().astype(np.uint8)
            # logger.info(img.dtype)
            # logger.info(img1.dtype)
            # logger.info(img1.shape)
            # logger.info(img.shape)
            # cv2.imwrite('hist_test_img.jpg',img)
            # cv2.imwrite('hist_test_img1.jpg',img1)
            img = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB)
            return np.mean(np.transpose(img, (2, 0, 1)), axis=(1, 2))
        else:
            raise NotImplementedError('not implet')

    def compare_histgram(self, h1, h2):
        # logger.info('h1')
        # logger.info(h1)
        # logger.info('h2')
        # logger.info(h2)
        return np.mean(np.abs(h1 - h2))

    def load_search_database(self, kinds):
        res = []
        for kind in kinds:
            cropus_datapath = osp.join('retrieval/cropus', kind, 'cropus1920.npy')
            assert osp.exists(cropus_datapath), 'No cropus'
            res.append(np.load(cropus_datapath))
        return res

    def load_search_index(self):
        p = osp.join('retrieval/cropus/index', self.first_class_id)
        cropus_datapath = osp.join(p, 'cropus.lst')

        def get_index(fn):
            with open(fn, 'r') as f:
                f_line = list(map(lambda x: x.strip(), f.readlines()))
            d = {}
            for idx, i in enumerate(f_line):
                d[i] = idx
            return d, f_line

        return get_index(cropus_datapath)

    def get_feature(self, img):
        if not isinstance(img, nd.NDArray):
            img = nd.array(img)
        cv2.imwrite('test.jpg', img.asnumpy())
        imgs = transform_test(img)
        fea = nd.mean(self.model(imgs), axis=0)
        '''
        fea = \
        logger.info('h1')
        logger.info(h1)
            nd.mean(nd.stack(*[self.model(transform_val(img, None)[0].expand_dims(axis=0)) for i in range(tt)]),
                    axis=0)[0]
        '''
        # print(np.sum(np.abs(fea.asnumpy())))
        # fea = self.pca.transform(fea.asnumpy()[np.newaxis,])

        # fea = self.pca.transform(fea.asnumpy().reshape(1,-1))
        # print(np.sum(np.abs(fea)))
        return fea.asnumpy().reshape(1, -1)

    def search_database(self, img, cropus_data, color_level, style_level):
        dof_threshold_config = {
            6: ([0.24, 0.26, 0.28], [6, 20], 2048, 64),
            4: ([0.18, 0.20, 0.22], [8, 10], 2048 * 4, 512)
        }
        threshold_styles, color_styles, c1, c2 = dof_threshold_config[int(self.first_class_id)]
        threshold_style = threshold_styles[style_level]
        color_style = color_styles[color_level]
        anchor = self.get_feature(img)
        anchor_hist = self.get_hist(img)
        fea_dist = np.sum((cropus_data - anchor) ** 2, axis=1)
        res = list(map(lambda x: x, filter(lambda x: fea_dist[x] < threshold_style, np.argsort(fea_dist)[:c1])))
        cropus_hists = self.cropus_hist[res]
        distance = np.arange(cropus_hists.shape[0]).astype(np.float32)
        for idx, i in enumerate(cropus_hists):
            distance[idx] = self.compare_histgram(anchor_hist, i)
        temp = []
        temp1 = []
        for idx, i in enumerate(res):
            if distance[idx] < color_style:
                temp.append(i)
                '''
                elif distance[idx] < 15:
                    temp1.append(i)
                '''
        res = list(map(lambda x: self.cropus_index_inverse[x], temp[:c2]))
        logger.info('fea dist')
        logger.info(list(map(lambda x: fea_dist[x], temp[:c2])))
        logger.info(sorted(fea_dist)[:c2])
        if len(res) == 0:
            logger.info('no result')
        elif len(res) < 10:
            logger.info('result counts less than 10')
        return res

    def get_mod(self, folder_name, checkpoint_name, ctx=mx.cpu()):
        net = symbol_factory.get_test_symbol(ctx=ctx)
        net.load_params(osp.join(folder_name, checkpoint_name), ctx=ctx)
        net.hybridize()
        return net


if __name__ == '__main__':
    model = Retrieval_model(ctx=mx.cpu())
    pairs_json = {}
    tic = time()
    # print(model.search('820115'))
    # imglist = read_to_list(test_txt_path)

    tic = time()
    fea1 = model.get_feature('../demo/2.png', tt=1)
    print(time() - tic)
    import time

    time.sleep(1000)
    '''
    print(nd.sum(fea1 == 0))
    fea2 = nd.stack(*[model.get_feature(osp.join('../demo', i)) for i in
            ['177838.jpg', '202027.jpg','377045.jpg','526853.jpg','img.jpeg','img1.jpg','1.png','img2.jpg','3.png']])
    print(nd.argsort(nd.sum((fea1 - fea2) ** 2,axis=1)))
    '''
