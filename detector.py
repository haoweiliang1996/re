from __future__ import print_function
from collections import namedtuple
import cv2

from timeit import default_timer as timer

import mxnet as mx
import numpy as np
import pandas as pd
import os.path as osp

from dataset.iterator import DetIter
from dataset.testdb import TestDB
from clothtools.logger import logger


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
        self.classname_en2id = pd.read_excel('/home/lhw/cloth/clothdata/class1and2.xlsx')

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
                time_elapsed = timer() - start
                if show_timer:
                    logger.info("Detection time for {} images: {:.4f} sec".format(
                        idx, time_elapsed))
            detections.append(pred[0].asnumpy())
        for output in detections:
            for i in range(output.shape[0]):
                det = output[i, :, :]
                res = det[np.where(det[:, 0] >= 0)[0]]
                result.append(res)
        return result

    def im_detect(self, im_list, root_dir=None, extension=None, show_timer=False):
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
        test_db = TestDB(im_list, root_dir=root_dir, extension=extension)
        test_iter = DetIter(test_db, 1, self.data_shape, self.mean_pixels,
                            is_train=False)
        return self.detect(test_iter, show_timer)

    def visualize_detection(self, pos, img):
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

    def visualize_detection(self, img, dets, classes=[], thresh=0.6):
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
        import matplotlib.pyplot as plt
        import random
        plt.imshow(img)
        height = img.shape[0]
        width = img.shape[1]
        colors = dict()
        for i in range(dets.shape[0]):
            cls_id = int(dets[i, 0])
            if cls_id >= 0:
                score = dets[i, 1]
                if score > thresh:
                    logger.info('get %s in pic'%cls_id)
                    if cls_id not in colors:
                        colors[cls_id] = (random.random(), random.random(), random.random())
                    xmin = int(dets[i, 2] * width)
                    ymin = int(dets[i, 3] * height)
                    xmax = int(dets[i, 4] * width)
                    ymax = int(dets[i, 5] * height)
                    rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                         ymax - ymin, fill=False,
                                         edgecolor=colors[cls_id],
                                         linewidth=3.5)
                    plt.gca().add_patch(rect)
                    class_name = str(cls_id)
                    if classes and len(classes) > cls_id:
                        class_name = classes[cls_id]
                    plt.gca().text(xmin, ymin - 2,
                                   '{:s} {:.3f}'.format(class_name, score),
                                   bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                                   fontsize=12, color='white')
        plt.show()

    def visualize_detection_to_onefile(self, img, dets, classes=[], thresh=0.6,img_name='test.jpg', output_dir='.'):
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
        import matplotlib.pyplot as plt
        import random
        plt.imshow(img)
        height = img.shape[0]
        width = img.shape[1]
        colors = dict()
        for i in range(dets.shape[0]):
            cls_id = int(dets[i, 0])
            if cls_id >= 0:
                score = dets[i, 1]
                if score > thresh:
                    logger.info('get %s in pic'%cls_id)
                    if cls_id not in colors:
                        colors[cls_id] = (random.random(), random.random(), random.random())
                    xmin = int(dets[i, 2] * width)
                    ymin = int(dets[i, 3] * height)
                    xmax = int(dets[i, 4] * width)
                    ymax = int(dets[i, 5] * height)
                    rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                         ymax - ymin, fill=False,
                                         edgecolor=colors[cls_id],
                                         linewidth=3.5)
                    plt.gca().add_patch(rect)
                    class_name = str(cls_id)
                    if classes and len(classes) > cls_id:
                        class_name = classes[cls_id]
                    plt.gca().text(xmin, ymin - 2,
                                   '{:s} {:.3f}'.format(class_name, score),
                                   bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                                   fontsize=12, color='white')
        #plt.show()
        plt.savefig(osp.join(output_dir,osp.basename(img_name)))
        plt.gcf().clear()
    def visualize_detection_tofile(self, img, dets, classes=[], thresh=0.6, img_name='test.jpg', output_dir='.'):
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

        height = img.shape[0]
        width = img.shape[1]
        for i in range(dets.shape[0]):
            cls_id = int(dets[i, 0])
            if cls_id >= 0:
                class1id = self.classname_en2id[self.classname_en2id.cnameen == classes[cls_id]].cid.iloc[0]
                score = dets[i, 1]
                if score > thresh:
                    xmin = int(dets[i, 2] * width)
                    ymin = int(dets[i, 3] * height)
                    xmax = int(dets[i, 4] * width)
                    ymax = int(dets[i, 5] * height)
                    img_out = img[ymin:ymax, xmin:xmax]
                    output_filename = osp.join(output_dir, str(class1id), img_name)
                    assert cv2.imwrite(output_filename, img_out)
                    '''
                    class_name = str(cls_id)
                    if classes and len(classes) > cls_id:
                        class_name = classes[cls_id]
                 '''
        '''
        import os.path as osp
        output_filename = osp.join(output_dir, img_name)
       '''

    def detect_and_visual(self, im_list, root_dir=None, extension=None,
                        classes=[], thresh=0.6, show_timer=False):
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

        dets = self.im_detect(im_list, root_dir, extension, show_timer=show_timer)
        logger.info('detecting ok,prepare to visual')
        if not isinstance(im_list, list):
            im_list = [im_list]
        assert len(dets) == len(im_list)
        for k, det in enumerate(dets):
            logger.info(osp.join(root_dir, im_list[k]))
            img = cv2.imread(osp.join(root_dir, im_list[k]))
            img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
            self.visualize_detection(img, det, classes, thresh)

    def detect_and_visual_to_onefile(self, im_list, root_dir=None, extension=None,
                        classes=[], thresh=0.6, show_timer=False,output_dir='./'):
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

        dets = self.im_detect(im_list, root_dir, extension, show_timer=show_timer)
        logger.info('detecting ok,prepare to visual to onefile')
        if not isinstance(im_list, list):
            im_list = [im_list]
        assert len(dets) == len(im_list)
        for k, det in enumerate(dets):
            logger.info(osp.join(root_dir, im_list[k]))
            img = cv2.imread(osp.join(root_dir, im_list[k]))
            img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
            self.visualize_detection_to_onefile(img, det, classes, thresh,output_dir=output_dir, img_name=im_list[k])

    def detect_and_save(self, im_list, root_dir=None, extension=None,
                        classes=[], thresh=0.6, show_timer=False, output_dir='.'):
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

        dets = self.im_detect(im_list, root_dir, extension, show_timer=show_timer)
        if not isinstance(im_list, list):
            im_list = [im_list]
        assert len(dets) == len(im_list)
        for k, det in enumerate(dets):
            logger.info(osp.join(root_dir, im_list[k]))
            img = cv2.imread(osp.join(root_dir, im_list[k]))
            # img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
            self.visualize_detection_tofile(img, det, classes, thresh, output_dir=output_dir, img_name=im_list[k])
    def detect_and_dump_res(self, im_list, root_dir=None, extension=None,
                        classes=[], thresh=0.6, show_timer=False,output_dir=''):
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

        dets = self.im_detect(im_list, root_dir, extension, show_timer=show_timer)
        logger.info('detecting ok,prepare to pickle')
        if not isinstance(im_list, list):
            im_list = [im_list]
        assert len(dets) == len(im_list)
        #det_res = namedtuple('det_res',['imgId','res'])
        import pandas as pd
        im_list_split = list(map(lambda x:x.split('/'),im_list))
        type_list = [i[0] for i in im_list_split]
        id_list = [i[-1] for i in im_list_split]
        res = pd.DataFrame({'type':type_list,'imgId':id_list,'res':dets})
        #res = pd.Panel({im_list[i].split('.')[0]:pd.DataFrame(dets[i],columns=['clsid','score','xmin','ymin','xmax','ymax']) for i in range(len(im_list))})
        import pickle
        with open(osp.join(output_dir,'ssd_res.pkl'),'wb') as f:
            pickle.dump(res,f)
