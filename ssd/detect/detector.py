from __future__ import print_function

import random
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np

from ssd.dataset.iterator import DetIter


class Detector(object):
    """
    SSD detector which hold a detection network and wraps detection API

    Parameters:
    ----------
    symbol : mx.Symbol
        detection network Symbol
    model_prefix : str
        name prefix of trained model
    epoch : int
        load epoch of trained model
    data_shape : int
        input data resize shape
    mean_pixels : tuple of float
        (mean_r, mean_g, mean_b)
    batch_size : int
        run detection with batch size
    ctx : mx.ctx
        device to use, if None, use mx.cpu() as default context
    """

    def __init__(self, symbol, model_prefix, epoch, data_shape, mean_pixels, \
                 batch_size=1, ctx=None):
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = mx.cpu()
        load_symbol, args, auxs = mx.model.load_checkpoint(model_prefix, epoch)
        if symbol is None:
            symbol = load_symbol
        self.mod = mx.mod.Module(symbol, label_names=None, context=ctx)
        self.data_shape = data_shape
        self.mod.bind(data_shapes=[('data', (batch_size, 3, data_shape[0], data_shape[1]))])
        self.mod.set_params(args, auxs)
        self.data_shape = data_shape
        self.mean_pixels = mean_pixels

    def detect(self, det_iter, show_timer=False):
        """
        detect all images in iterator

        Parameters:
        ----------
        det_iter : DetIter
            iterator for all testing images
        show_timer : Boolean
            whether to print out detection exec time

        Returns:
        ----------
        list of detection results
        """
        num_images = det_iter._size
        result = []
        detections = []
        if not isinstance(det_iter, mx.io.PrefetchingIter):
            det_iter = mx.io.PrefetchingIter(det_iter)
        start = timer()
        for idx, (pred, _, _) in enumerate(self.mod.iter_predict(det_iter)):
            if idx % 1000 == 0:
                if show_timer:
                    print(idx)
            detections.append(pred[0].asnumpy())
        time_elapsed = timer() - start
        if show_timer:
            print("Detection time for {} images: {:.4f} sec".format(
                num_images, time_elapsed))
        for output in detections:
            for i in range(output.shape[0]):
                det = output[i, :, :]
                res = det[np.where(det[:, 0] >= 0)[0]]
                result.append(res)
        return result

    def im_detect(self, img, root_dir=None, extension=None, show_timer=False):
        """
        wrapper for detecting multiple images

        Parameters:
        ----------
        im_list : list of str
            image path or list of image paths
        root_dir : str
            directory of input images, optional if image path already
            has full directory information
        extension : str
            image extension, eg. ".jpg", optional

        Returns:
        ----------
        list of detection results in format [det0, det1...], det is in
        format np.array([id, score, xmin, ymin, xmax, ymax]...)
        """
        test_iter = DetIter(img, 1, self.data_shape, self.mean_pixels,
                            is_train=False)
        return self.detect(test_iter, show_timer)

    def visualize_detection_to_memory(self, img, dets, classes=[], thresh=0.6, img_name='test.jpg', output_dir='.'):
        """
        visualize detections in one image

        Parameters:
        ----------
        img : numpy.array
            image, in bgr format
        dets : numpy.array
            ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
            each row is one object
        classes : tuple or list of str
            class names
        thresh : float
            score threshold
        """
        raise NotImplementedError('not imple')

    def detect_and_return(self, img,
                          classes=['skirt', ], thresh=0.6, show_timer=False):
        """
        wrapper for im_detect and visualize_detection

        Parameters:
        ----------
        im_list : list of str or str
            image path or list of image paths
        root_dir : str or None
            directory of input images, optional if image path already
            has full directory information
        extension : str or None
            image extension, eg. ".jpg", optional

        Returns:
        ----------

        """
        dets = self.im_detect(img, show_timer=show_timer)
        for k, det in enumerate(dets):
            # img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
            return self.visualize_detection_to_memory(img, det, classes, thresh)

    def visualize_detection_matplot(self, pos, img):
        """
        visualize detections in one image

        Parameters:
        ----------
        img : numpy.array
            image, in bgr format
        dets : numpy.array
            ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
            each row is one object
        classes : tuple or list of str
            class names
        thresh : float
            score threshold
        """
        if not isinstance(pos,list):
            pos = [pos,]
        plt.imshow(img)
        height = img.shape[0]
        width = img.shape[1]
        colors = dict()
        colors[0] = (random.random(), random.random(), random.random())
        cls_id = 0
        for det in pos:
            score = det[0]
            xmin = int(det[1] * width)
            ymin = int(det[2] * height)
            xmax = int(det[3] * width)
            ymax = int(det[4] * height)
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=3.5)
            plt.gca().add_patch(rect)
            class_name = str(cls_id)
            plt.gca().text(xmin, ymin - 2,
                           '{:s} {:.3f}'.format(class_name, score),
                           bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                           fontsize=12, color='white')
        plt.show()
