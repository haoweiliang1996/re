import os
import sys

from logger import logger

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, '../../'))
import os.path as osp

# define a simple data batch

import mxnet as mx
import matplotlib.pyplot as plt
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
        self.ctx = ctx
        self.model = self.get_mod(
            folder_name=osp.join(curr_path, '../../', 'retrieval/checkpoint/%s' % self.first_class_id),
            checkpoint_name='net_best.params', ctx=ctx)
        self.database = self.load_search_database(
            [osp.join('database', self.first_class_id), ])

        (self.cropus_index, self.cropus_index_inverse) = self.load_search_index()
        self.cropus_hist = self.load_hist_database()

    def load_hist_database(self):
        base = osp.join(curr_path, '../../', 'retrieval/cropus/hist', self.first_class_id)
        hist_path = osp.join(base, 'cropus_hist.npy')
        assert osp.exists(hist_path), 'hist %s not found' % hist_path
        return np.load(hist_path)

    def get_hist(self, img, **kwargs):
        if not isinstance(img, nd.NDArray):
            img1 = nd.array(img)
        else:
            img1 = img
        if int(self.first_class_id) != 4:
            img1 = mx.img.resize_short(img1, 224)
            img1 = mx.img.center_crop(img1, (112, 112))[0].asnumpy().astype(np.uint8)
            img = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB)
            return np.mean(np.transpose(img, (2, 0, 1)), axis=(1, 2))
        else:
            c_detector = kwargs['color_detector']
            tic = time()
            pos, imgs = c_detector.detect_and_return(img1, thresh=0.24)
            logger.info('use %f time to detect color' % (time() - tic))
            if kwargs.get('debug') is not None:
                c_detector.visualize_detection_matplot(pos, img1)
            res = []
            for i in range(len(imgs)):
                if kwargs.get('debug') is not None:
                    img = imgs[i].astype(np.uint8)
                    plt.imshow(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)  # 在cvtColor前要先变成uint8
                m = np.mean(np.transpose(img, (2, 0, 1)), axis=(1, 2))
                res.append(m)
            return np.mean(res, axis=0)

    def compare_histgram(self, h1, h2):
        # logger.info('h1')
        # logger.info(h1)
        # logger.info('h2')
        # logger.info(h2)
        return np.mean(np.abs(h1 - h2))

    # @lru_cache(maxsize=4)
    def load_search_database(self, kinds):
        res = []
        for kind in kinds:
            cropus_datapath = osp.join(curr_path, '../../', 'retrieval/cropus', kind, 'cropus1920.npy')
            assert osp.exists(cropus_datapath), 'No cropus'
            res.append(np.load(cropus_datapath))
        return res

    def load_search_index(self):
        p = osp.join(curr_path, '../../', 'retrieval/cropus/index', self.first_class_id)
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

    def search_database(self, img, cropus_data, color_level, style_level, **kwargs):
        dof_threshold_config = {
            6: ([0.24, 0.26, 0.28], [6, 20], 2048, 64),
            4: ([0.20, 0.21, 0.22], [8, 10], 2048 * 4, 512),
            5: ([0.20, 0.21, 0.22], [8, 10], 2048 * 4, 512),
            7: ([0.20, 0.21, 0.22], [8, 10], 2048 * 4, 512)
        }
        threshold_styles, color_styles, c1, c2 = dof_threshold_config[int(self.first_class_id)]
        threshold_style = threshold_styles[style_level]
        color_style = color_styles[color_level]
        anchor = self.get_feature(img)
        if int(self.first_class_id) == 4:
            anchor_hist = self.get_hist(img, color_detector=kwargs['color_detector'])
        else:
            anchor_hist = self.get_hist(img)
        tic = time()
        fea_dist = np.sum((cropus_data - anchor) ** 2, axis=1)
        logger.info('use %s second to cal distance' % (time() - tic))
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
        logger.info(list(map(lambda x: fea_dist[x], temp[:c2]))[:16])
        # logger.info(sorted(fea_dist)[:c2])
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
    from ssd.detect.color_detector import ColorDetector
    from ssd import demo

    color_detector = demo.get_ssd_model(detector_class=ColorDetector,
                                        prefix=os.path.join(curr_path, '../../ssd', 'checkpoint', 'maincolor', 'ssd'),
                                        ctx=mx.cpu())
    img = cv2.cvtColor(cv2.imread(os.path.join('/Users/haowei/Downloads',
                                               '37245de15c3e400b8e30b61c32422f7a.jpg')),
                       cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.show()
    for first_class_id in [4, ]:
        model = Retrieval_model(ctx=mx.cpu(), first_class_id=first_class_id)
        pairs_json = {}
        tic = time()
        res = model.search_database(img, model.database[0], 0, 0, color_detector=color_detector)
        print(res[:8])
        # print(model.search('820115'))
        # imglist = read_to_list(test_txt_path)

        # fea1 = model.get_feature(img)
        print(time() - tic)

    # time.sleep(1000)
    '''
    print(nd.sum(fea1 == 0))
    fea2 = nd.stack(*[model.get_feature(osp.join('../demo', i)) for i in
            ['177838.jpg', '202027.jpg','377045.jpg','526853.jpg','img.jpeg','img1.jpg','1.png','img2.jpg','3.png']])
    print(nd.argsort(nd.sum((fea1 - fea2) ** 2,axis=1)))
    '''
