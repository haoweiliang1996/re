import os
import sys
import mxnet as mx
import numpy as np
import os
import sys

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, '..'))

from symbol_factory import multiTaskClassifyNetwork,multiClassifyLoss,evaluate_accuracy,validate
from mxnet import gluon, image, init, nd
from dataset.lstDataset import lstDataset

ctx = [mx.gpu(i) for i in range(2)]
#ctx = [mx.cpu(),]

def transform_train(data, label):
    im = data
    data_width = 224
    auglist = image.CreateAugmenter(data_shape=(3, data_width, data_width))
    for aug in auglist:
        im = aug(im)
    im = im.astype('float32') / 255
    im = nd.transpose(im, (2,0,1))
    im = mx.nd.image.normalize(im, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return (im,label)

def transform_val(data, label):
    resize_short_width = 256
    data_width = 224
    im = data.astype('float32') / 255
    im = image.resize_short(im, resize_short_width)
    im, _ = image.center_crop(im, (data_width, data_width))
    im = nd.transpose(im, (2,0,1))
    im = mx.nd.image.normalize(im, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return (im, label)

def unit_test_getLabelAndPredict():
    from dataset.clothall import get_label_index
    select_index,lenof_labels = get_label_index(dataset_part='cloth4'
                                            ,use_color=True)
    print('select_index,lenof_labels')
    print(select_index,lenof_labels)
    net = multiTaskClassifyNetwork(lenof_labels=lenof_labels,ctx=ctx)
    net.hybridize()
    net.output.initialize(init.Xavier(), ctx = ctx)
    net.load_params('/home/lhw/cloth/mxnet-classify/checkpoint/clothall/cloth4c/densenet201/net_best.params',ctx=ctx)
    print('load params')
    # Define DataLoader
    train_dataset = lstDataset(pic_root='/home/lhw/cloth/mxnet-classify/data/CLOTHdevkit/clothall',
            items_filename = '/home/lhw/cloth/mxnet-classify/data/train_meta/clothall/cloth4c/all.lst',
                            transform=transform_val)
    print(len(train_dataset))
    batch_size = 1024
    num_workers = 4
    train_data = gluon.data.DataLoader(
        train_dataset,batch_size=batch_size, shuffle=True,
        num_workers=num_workers, last_batch='keep')
    '''
    (d,l) = next(iter(train_data))
    data = gluon.utils.split_and_load(d, ctx, even_split=True)
    label = gluon.utils.split_and_load(l, ctx, even_split=True)

    acc = [evaluate_accuracy(Y,preds=net(X),select_index=select_index) for X, Y in zip(data, label)]
    '''
    acc = validate(net,train_data,ctx=ctx,select_index = select_index)

    #print(d,l)
    print(acc)




def unit_test():
    lenof_labels = [3,4]
    net = multiTaskClassifyNetwork(lenof_labels=lenof_labels,ctx=ctx)
    net.hybridize()
    net.output.initialize(init.Xavier(), ctx = ctx)
    net1 = multiTaskClassifyNetwork(lenof_labels=3,ctx= ctx)
    net1.hybridize()
    net1.output.initialize(init.Xavier(), ctx = ctx)
    #print(len(net(imgs)))
    print(net(imgs))
    print(net1(imgs))
    o = net(imgs)
    o1 = net(imgs)
    print('type of feature part',type(net.features))
    print('type of output part',type(net.output))
    print(net1.features(imgs))
    L = multiClassifyLoss(lenof_labels,select_index=[0,1])
    labels = nd.random.normal(shape=[2,2],ctx=mx.gpu(0))
    print(L(labels,o))
    L = multiClassifyLoss(lenof_labels=[3,],select_index=[0])
    labels = nd.random.normal(shape=[2,2],ctx=mx.gpu(0))
    print(L(labels,o))


unit_test_getLabelAndPredict()
#unit_test()
