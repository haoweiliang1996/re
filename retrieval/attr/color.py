import os
import sys

import cv2
import mxnet as mx
import numpy as np

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, '../..'))
from mxnet import nd

from retrieval.dataloader.transform import transform_val
from retrieval.mxsymbol.symbol_factory import multiTaskClassifyNetwork
from logger import logger
from time import time


class ColorClassifier():
    def __init__(self, ctx):
        self.lenof_labels = [17, 17]
        net = multiTaskClassifyNetwork(lenof_labels=self.lenof_labels, ctx=ctx)
        net.hybridize()
        params_path = os.path.join(curr_path, 'checkpoint', 'color', 'net_best.params')
        net.load_params(params_path, ctx=ctx)
        self.net = net
        self.ctx = ctx

    def predict(self, img):
        tic = time()
        if not isinstance(img, nd.NDArray):
            img = nd.array(img)
        img = nd.array(img)
        img, _ = transform_val(img, None)

        img = nd.stack(*[img,])
        outputs = self.net(img)

        #todo asnumpy is very slow ,cost about 0.2 second
        outputs = [nd.softmax(i, axis=1).asnumpy() for i in outputs]

        preds = []
        index2label = [0, 4096, 2048
            , 128
            , 8192
            , 4
            , 16384
            , 64
            , 256
            , 65536
            , 8
            , 32
            , 2
            , 32768
            , 16
            , 512
            , 1024]
        for k in range(len(self.lenof_labels)):
            # pdb.set_trace()
            sort_index = np.argsort(outputs[k], axis=1)[0][::-1].astype(np.int32)
            res = []
            for idx in sort_index:
                i = int(sort_index[idx])
                if outputs[k][0][i] >= 0.2:
                    res.append(index2label[i])
            res = res[:3]
            preds.append(res)
        temp = []
        for i in preds:
            temp += i
        flatten = np.array(temp).flatten()
        cc = 0
        for i in flatten:
            if int(i) == 0:
                cc += 1
        if cc > 2:
            logger.info('zero in color more than two')
            raise RuntimeError('zero in color more than two')
        if preds[0][0] == 0 and preds[1][0] == 0:
            flag = 0
        else:
            flag = int(np.sum(np.unique(flatten)))
        logger.info('flag :%s, preds %s' % (flag, str(preds)))
        logger.info('color classifyier use time %s'%(time()-tic))
        return flag


if __name__ == '__main__':
    model = ColorClassifier(ctx=mx.cpu())
    print(model.predict(cv2.imread(os.path.join('/Users/haowei/Downloads', '55a688bd-a908-4c50-a69e-2425c7802702.jpg'))))
