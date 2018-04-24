import numpy as np
import mxnet as mx
from ssd.detect.detector import Detector


class ColorDetector(Detector):
    def visualize_detection_to_memory(self, img, dets, classes=[], thresh=0.6, img_name='test.jpg', output_dir='.'):
        """
        visualize detections to memory

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
        res = []
        height = img.shape[0]
        width = img.shape[1]
        for i in range(dets.shape[0]):
            cls_id = int(dets[i, 0])
            if cls_id >= 0:
                score = dets[i, 1]
                if score > thresh:
                    xmin = int(dets[i, 2] * width)
                    ymin = int(dets[i, 3] * height)
                    xmax = int(dets[i, 4] * width)
                    ymax = int(dets[i, 5] * height)
                    img_out = img[ymin:ymax, xmin:xmax]
                    res.append(img_out)
        if len(res) == 0:
            res = [mx.image.CenterCropAug((112, 112))(mx.nd.array(img)).asnumpy(),]
        return None,res
