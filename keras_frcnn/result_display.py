# encoding:utf-8
import numpy as np
from keras_frcnn import network_aux_roi
import cv2
class Display():
    def __init__(self,bboxes,probs,ratio):
        self.bboxes = bboxes
        self.probs = probs
        self.all_dets = []
        self.ratio = ratio
    # 得到box在原图的坐标(x1, y1, x2, y2)
    def get_real_coordinates(self, bbox):
        r_bbox = []       
        r_bbox.append(int(round(bbox[0] // self.ratio)))
        r_bbox.append(int(round(bbox[1] // self.ratio)))
        r_bbox.append(int(round(bbox[2] // self.ratio)))
        r_bbox.append(int(round(bbox[3] // self.ratio)))
        return r_bbox
    
    def result(self,):
        for key in self.bboxes:
            bbox = np.array(self.bboxes[key])
            probs = np.array(self.probs[key])
            new_boxes, new_probs = network_aux_roi.non_max_suppression_fast(bbox, probs, overlap_thresh=0.5)
            for jk in range(new_boxes.shape[0]):
                r_bbox = self.get_real_coordinates(new_boxes[jk,:])
                self.all_dets.append((key,r_bbox,100*new_probs[jk]))
        print self.all_dets

            #return self.all_dets
    
    def draw(self, C, img):
        class_mapping = C.class_mapping
        class_mapping = {v: k for k, v in class_mapping.items()}
        class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

        for box in self.all_dets:
            key = box[0]
            r_bbox = box[1]
            prob = box[2]
            cv2.rectangle(img,(r_bbox[0], r_bbox[1]), (r_bbox[2], r_bbox[3]), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)
            textLabel = '{}: {:.2f}'.format(key,prob)
            (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
            textOrg = (r_bbox[0], r_bbox[1] - 0)
            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5 + 30), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5 + 30), (0, 0, 0), 2)
            cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5 +30), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5 +30), (255, 255, 255), -1)
            cv2.putText(img, textLabel, (r_bbox[0], r_bbox[1]-0+30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
            cv2.imwrite('./results_imgs/predict.png',img)