import os
import sys

import cv2
import matplotlib.pyplot as plt
import mxnet as mx

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, '../..'))
from ssd.detect.detector import Detector
from ssd import demo


# Todo 这里和试验的时候不一致
class ColorDetector(Detector):
    def visualize_detection_to_memory(self, img, dets, classes=[], thresh=0.25, img_name='test.jpg', output_dir='.',
                                      show_img=False):
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
        position = []
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
                    position.append(dets[i, 1:])
                    res.append(img_out)
                    if len(res) == 3:
                        break
        if len(res) == 0:
            res = [mx.image.CenterCropAug((112, 112))(mx.nd.array(img)).asnumpy(), ]
        return position, res


if __name__ == '__main__':
    model = demo.get_ssd_model(detector_class=ColorDetector,
                               prefix=os.path.join(curr_path, '../', 'checkpoint', 'maincolor', 'ssd'), ctx=mx.cpu())

    #img = cv2.imread(os.path.join('/Users/haowei/Downloads', '37245de15c3e400b8e30b61c32422f7a.jpg'))
    img = cv2.cvtColor(cv2.imread(os.path.join('/Users/haowei/Downloads',
                                               'mmexport1524744468360.jpg')),
                       cv2.COLOR_BGR2RGB)
    pos, imgs = model.detect_and_return(img, thresh=0.24)
    model.visualize_detection_matplot(pos, img)
    for i in imgs:
        plt.imshow(i)
        plt.show()
