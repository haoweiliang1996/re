import os
import sys

import mxnet as mx
import numpy as np

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, '..'))

from mxnet import gluon, metric, nd
from sklearn.metrics import classification_report, confusion_matrix


def validate(net, val_data, ctx, select_index):
    metrics = [mx.metric.Accuracy() for i in range(len(select_index))]
    topk3_metrics = [mx.metric.TopKAccuracy(top_k=3) for i in range(len(select_index))]
    topk2_metrics = [mx.metric.TopKAccuracy(top_k=2) for i in range(len(select_index))]
    preds = [np.array([]) for i in range(len(select_index))]
    labels = [np.array([]) for i in range(len(select_index))]
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        label = [onelabel.T[select_index,] for onelabel in label]
        outputs = [net(X) for X in data]
        for j in range(len(outputs)):
            for k in range(len(select_index)):
                # pdb.set_trace()
                preds[k] = np.concatenate((preds[k], nd.argmax(outputs[j][k], axis=1).asnumpy()))
                labels[k] = np.concatenate((labels[k], label[j][k].asnumpy()))
                metrics[k].update([label[j][k], ], [outputs[j][k], ])
                topk2_metrics[k].update([label[j][k], ], [outputs[j][k], ])
                topk3_metrics[k].update([label[j][k], ], [outputs[j][k], ])
    val_accs = []
    for i in range(len(metrics)):
        _, acc = metrics[i].get()
        val_accs.append(acc)
        _, acc = topk2_metrics[i].get()
        val_accs.append(acc)
        _, acc = topk3_metrics[i].get()
        val_accs.append(acc)
    for i in range(len(select_index)):
        print(classification_report(labels[i], preds[i]))
        cm = confusion_matrix(labels[i], preds[i])
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(cm)
    # val_accs = (metric.get() for metric in metrics)
    return ((val_accs, 0 / len(val_data)))


def _get_symbol(model_name, ctx=mx.cpu(), pretrained=False):
    finetune_net = gluon.model_zoo.vision.get_model(model_name, pretrained=pretrained,ctx=ctx).features
    return finetune_net


class _MultiClassifyTask(gluon.block.HybridBlock):
    def __init__(self, lenof_labels, **kwargs):
        super(_MultiClassifyTask, self).__init__(**kwargs)
        if isinstance(lenof_labels, int):
            lenof_labels = (lenof_labels,)
        with self.name_scope():
            for i in range(len(lenof_labels)):
                exec('self.fc%s = gluon.nn.Dense(%s)' % (i, lenof_labels[i]))
        self.forward_script = ','.join(['self.fc%s(x)' % i for i in range(len(lenof_labels))])

    def hybrid_forward(self, F, x):
        # return self.fc1(x), self.fc2(x), self.fc3(x), self.fc4(x), self.fc5(x), self.fc6(x),self.fc_color(x) #, F.split(self.fc_color(x),num_outputs=248, axis=1)
        return eval(self.forward_script)


softmax_cross_entropys = gluon.loss.SoftmaxCrossEntropyLoss()
softmax_cross_entropys.hybridize()


class multiClassifyLoss(gluon.loss.Loss):
    def __init__(self, lenof_labels, select_index=None, weight=None, batch_axis=0, **kwargs):
        super(multiClassifyLoss, self).__init__(weight, batch_axis, **kwargs)
        self.lenof_labels = lenof_labels
        self.select_index = select_index

    def hybrid_forward(self, F, label, pred):
        if self.select_index is not None:
            label = label.T[self.select_index,]
        else:
            label = label.T
        return F.sum(F.stack(*[softmax_cross_entropys(pred[i], label[i]) \
                               for i in range(len(self.lenof_labels))]), axis=0)


class multiTaskClassifyNetwork(gluon.HybridBlock):
    def __init__(self, lenof_labels, ctx, model_name='densenet201', **kwargs):
        super(multiTaskClassifyNetwork, self).__init__(**kwargs)
        with self.name_scope():
            self.features = _get_symbol(model_name, ctx=ctx, pretrained=True)
            self.output = _MultiClassifyTask(lenof_labels)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


'''
def get_train_symbol(lenof_labels,ctx=mx.cpu(), num_classes=1):
    net = gluon.nn.HybridSequential()
    body = __get_symbol(net_work,ctx=ctx,pretrained=True)
    body.hybridize()
    net.collect_params().reset_ctx(ctx)
    output = MultiClassifyTask(lenof_labels)
    output.collect_params().initialize(mx.init.Xavier(),ctx=ctx)
    output.hybridize()
    with net.name_scope():
        net.add(body)
        net.add(output)
    return net
'''


def get_test_symbol(net_work='densenet201', ctx=mx.cpu()):
    net = _get_symbol(net_work, ctx)

    class L2Normalization(gluon.HybridBlock):
        def hybrid_forward(self, F, x):
            return F.L2Normalization(x, mode='instance')

    with net.name_scope():
        net.add(L2Normalization(prefix=net.prefix))
    net.hybridize()
    return net
