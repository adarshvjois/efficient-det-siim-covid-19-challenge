import io
from torch.utils.data.dataset import Dataset
import glob
import os
import cv2 as cv
import numpy as np
EPS = 1e-8

class DataSetLoader(Dataset):
    def __init__(self,folder,resize,original_size=600,iou_thresh=0.3,sequence_lenght=10,eps: float = EPS) -> None:
        super().__init__()
        self.eps = eps
        self.iou_thresh = iou_thresh
        self.resize = resize
        self.original_size = original_size
        self.image_path = sorted(glob.glob(os.path.join(folder,"images","*.jpeg"))) 
        self.label_path = sorted(glob.glob(os.path.join(folder,"labels","*.txt")))
        self.buffer_labels = []
        self.buffer_images = []
        self.sequence_lenght = sequence_lenght

    def _process_image(self,path):
        image = cv.imread(path)
        image = cv.resize(image,self.resize)
        return image
    
    def _process_labels(self,path):
        print(path)
        bboxes = []
        classes = []
        with open(path) as fin:
            for line in fin:
                x = line.strip().split(",")
                left = (float(x[0])*self.resize[0]/self.original_size)
                top = (float(x[3])*self.resize[0]/self.original_size)
                right = (float(x[2])*self.resize[0]/self.original_size)
                bottom = (float(x[1])*self.resize[0]/self.original_size)
                box = [left,top,right,bottom]
                if(len(box)>0):
                    bboxes.append(box)
                    classes.append(1)
        return self._assign_labels_to_anchors(bboxes,classes,self._create_anchor_boxes())

    def _bb_intersection_over_union(self,gt_box, pred_box):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(gt_box[0], pred_box[0])
        yA = max(gt_box[1], pred_box[1])
        xB = min(gt_box[2], pred_box[2])
        yB = min(gt_box[3], pred_box[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)
        boxBArea = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    def _assign_labels_to_anchors(self,gt_boxes,gt_classes,anchors):
        #gt_boxes [left,top,right,bottom] Number_of_objectsx4
        #gt_classes Number_of_objectsx1
        #anchors NxAx4 N = feat_map_size**2
        target,indices = self._find_indices_iou(gt_boxes,anchors)
        return target,indices
    
    def _find_indices_iou(self,gt_boxes,anchors):
        indices_of_matched_boxes = []
        target = []
        for gt_box in gt_boxes:
            for outer,box in enumerate(anchors):
                for inner,bb in enumerate(box):
                    if(self._bb_intersection_over_union(gt_box,bb)>self.iou_thresh):
                        target.append(self._get_normalized_target(gt_box,bb))
                        indices_of_matched_boxes.append([outer,inner])
        return target,indices_of_matched_boxes

    def _get_normalized_target(self,gt,pred):
        xcenter,ycenter, h, w = self._xy_to_center(gt)
        xcenter_a,ycenter_a, ha, wa = self._xy_to_center(pred)
        ha += self.eps
        wa += self.eps
        h += self.eps
        w += self.eps
        tx = (xcenter - xcenter_a) / wa
        ty = (ycenter - ycenter_a) / ha
        tw = np.log(w / wa)
        th = np.log(h / ha)
        return [tx, ty, tw, th]
    
    def _xy_to_center(self,box):
        width = float(box[2]-box[0])
        height = float(box[3]-box[1])
        center_x = box[0]+width/2
        center_y = box[1]+height/2
        return[center_x,center_y,width,height]

    def _create_anchor_boxes(self):
        boxes_level = []
        boxes_all = []
        division_factor = 10
        anchors = [self.resize[0]/12,self.resize[0]/16,self.resize[0]/32]
        scaled_w = self.resize[0]/division_factor
        scaled_h = self.resize[1]/division_factor
        for anchor in anchors:
            anchor_size_x_2 = anchor
            anchor_size_y_2 = anchor
            stride = [scaled_w/2,scaled_h/2]
            x = np.arange(stride[1] / 2, self.resize[1], stride[1])
            y = np.arange(stride[0] / 2, self.resize[0], stride[0])    
            xv, yv = np.meshgrid(x, y)
            xv = xv.reshape(-1)
            yv = yv.reshape(-1)
            boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                                yv + anchor_size_y_2, xv + anchor_size_x_2))
            boxes = np.swapaxes(boxes, 0, 1)
            boxes_level.append(np.expand_dims(boxes, axis=1))
        boxes_level = np.concatenate(boxes_level, axis=1)
        return boxes_level

    def __getitem__(self, index):

        if(index==0):
            for i in range(self.sequence_lenght):
                targets,indices = self._process_labels(self.label_path[i])
                image = self._process_image(self.image_path[i])
                self.buffer_labels.append([targets,indices])
                self.buffer_images.append(image)
        else:
            targets,indices = self._process_labels(self.label_path[index+self.sequence_lenght-1])
            image = self._process_image(self.image_path[index+self.sequence_lenght-1])
            self.buffer_labels = self.buffer_labels[1:] #removing the first item
            self.buffer_labels.append([targets,indices])
            self.buffer_images = self.buffer_images[1:] 
            self.buffer_images.append(image)
        return self.buffer_images,self.buffer_labels

    def _draw_anchor_boxes(self,boxes):
        img_1 = np.zeros([self.resize[0]*2,self.resize[1]*2,3],dtype=np.uint8)
        img_1.fill(255)
        start_point = (int(self.resize[0]/2),int(self.resize[0]/2))
        end_point = (int(self.resize[0]*3/2),int(self.resize[0]*3/2))
        image = cv.rectangle(img_1,start_point , end_point, (125,0,0), 10)
        for scale_index,box in enumerate(boxes):
            for ind,bb in enumerate(box):
                anchor_top_left = (int(bb[0]+self.resize[0]/2),int(bb[1]+self.resize[0]/2))
                anchor_bottom_right = (int(bb[2]+self.resize[0]/2),int(bb[3]+self.resize[0]/2))
                image = cv.rectangle(image,anchor_top_left , anchor_bottom_right, (scale_index*2,0,255), 1)
            break
        cv.imwrite("test.jpg",image)

if __name__ == "__main__":
    ds = DataSetLoader("C:\\Users\\Kashyap\\bkp\\source\\repos\\efficient-det-siim-covid-19-challenge\\transformer\\BouncingBalls\\train",[300,300])
    dataloader_iter = iter(ds)
    try:
        while True:
            x,y = next(dataloader_iter)
            print(len(x),len(y))
    except StopIteration:
        pass
