import os
import sys

import cv2
import matplotlib.pyplot as plt
import mxnet as mx

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, '../..'))
from ssd.detect.detector import Detector
from ssd import demo


class SingleItemDetector(Detector):
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
        height = img.shape[0]
        width = img.shape[1]
        for i in range(min(1, dets.shape[0])):  # range(dets.shape[0]):
            cls_id = int(dets[i, 0])
            if cls_id >= 0:
                score = dets[i, 1]
                if score > thresh:
                    xmin = int(dets[i, 2] * width)
                    ymin = int(dets[i, 3] * height)
                    xmax = int(dets[i, 4] * width)
                    ymax = int(dets[i, 5] * height)
                    img_out = img[ymin:ymax, xmin:xmax]
                    return dets[i, 1:], img_out
        return None, None

if __name__ == '__main__':
    model = demo.get_ssd_model(detector_class=SingleItemDetector,
                               prefix=os.path.join(curr_path, '../', 'checkpoint', 'type3', 'ssd'), ctx=mx.cpu())

    img = cv2.imread(os.path.join('/Users/haowei/Downloads', 'mmexport1524748553162.jpg'))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    pos, imgs = model.detect_and_return(img, thresh=0.1)
    model.visualize_detection_matplot(pos, img)
    if not isinstance(imgs,list):
        imgs = [imgs,]
    for i in imgs:
        plt.imshow(i)
        plt.show()
    from retrieval.attr.color import ColorClassifier
    model = ColorClassifier(ctx=mx.cpu())
    print(model.predict(imgs[0]))
    from ssd.detect.color_detector import ColorDetector

    model = demo.get_ssd_model(detector_class=ColorDetector,
                               prefix=os.path.join(curr_path, '../', 'checkpoint', 'maincolor', 'ssd'), ctx=mx.cpu())

    img = imgs[0]

    pos, imgs = model.detect_and_return(img, thresh=0.24)
    model.visualize_detection_matplot(pos, img)
    for i in imgs:
        plt.imshow(i)
        plt.show()

