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


class Color_model():
    def __init__(self, ctx):
        self.lenof_labels = [17, 17]
        net = multiTaskClassifyNetwork(lenof_labels=self.lenof_labels, ctx=ctx)
        net.hybridize()
        params_path = os.path.join(curr_path, 'checkpoint', 'color', 'net_best.params')
        net.load_params(params_path, ctx=ctx)
        self.net = net
        self.ctx = ctx

    def predict(self, img):
        if not isinstance(img, nd.NDArray):
            img = nd.array(img)
        img = nd.array(img)
        img, _ = transform_val(img, None)

        outputs = self.net(nd.array(img.asnumpy()[np.newaxis, :], ctx=self.ctx))
        outputs = [nd.softmax(i, axis=1) for i in outputs]

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
            sort_index = nd.argsort(outputs[k], axis=1)[0].asnumpy()[::-1].astype(np.int32)
            res = []
            for i in sort_index:
                i = int(i)
                if outputs[k][0][i] >= 0.1:
                    res.append(index2label[i])
            res = res[:3]
            preds.append(res)
        return preds


if __name__ == '__main__':
    model = Color_model(ctx=mx.cpu())
    print(model.predict(cv2.imread(os.path.join('/Users/haowei/Downloads', '00_166874_middlel.jpg'))))
