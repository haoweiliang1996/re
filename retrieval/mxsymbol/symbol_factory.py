import mxnet as mx
from mxnet import gluon
from os.path import join


def __get_symbol(ctx=mx.cpu()):
    return gluon.model_zoo.vision.densenet201(pretrained=True, ctx=ctx).features


def get_train_symbol(ctx=mx.cpu(), num_classes=1):
    pass


def get_test_symbol(ctx=mx.cpu()):
    net = __get_symbol(ctx)

    class L2Normalization(gluon.HybridBlock):
        def hybrid_forward(self, F, x):
            return F.L2Normalization(x, mode='instance')

    with net.name_scope():
        net.add(L2Normalization(prefix=net.prefix))
    return net
