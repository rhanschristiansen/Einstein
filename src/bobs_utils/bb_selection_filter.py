from sympy.geometry import Polygon,Point
import numpy as np

# Selector: select one from a list
# Filter: pass or none pass

class BB_Selector:
    def __init__(self,img):
        self.height, self.width, _ = img.shape
        # print(self.width, self.height)

    def select(self, bb_list):
        return None


class Trapezoid_Selector(BB_Selector):
    def select(self,bb_list):
        p1 = (self.width*2/5,0)
        p2 = (self.width*3/5,0)
        p3 = (self.width,self.height)
        p4 = (0,self.height)
        valid_region = Polygon(p1,p2,p3,p4)

        selectedBox = np.asarray([-1, -1, -1, -1])
        temp = 0
        for bbox in bb_list:
            bboxCenter = Point((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            if valid_region.encloses_point(bboxCenter):     # bbox center lies in the trapezoid region
                # select bbox closest to bottom
                if bboxCenter.y > temp:
                    temp = bboxCenter.y
                    selectedBox = bbox

        return selectedBox


# select only the center triangle region
class Triangle_Selector(BB_Selector):
    def select(self, bb_list):
        p1 = (self.width * 1 / 2, 0)
        p2 = (self.width, self.height)
        p3 = (0, self.height)
        filter = Triangle_Filter(p1,p2,p3)
        selectedBox = np.asarray([-1, -1, -1, -1])
        temp = 0
        for bbox in bb_list:
            bboxCenter = Point((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            if  filter.filter(bboxCenter):    # bbox center lies in the trapezoid region
                # select bbox closest to bottom
                if bboxCenter.y > temp:
                    temp = bboxCenter.y
                    selectedBox = bbox

        return selectedBox




class Triangle_Filter:
    def __init__(self,p1,p2,p3):
        self.valid_region = Polygon(p1, p2, p3)

    def filter(self,in_point):
        return self.valid_region.encloses_point(in_point)