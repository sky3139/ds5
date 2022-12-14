

import pickle
import cv2
import numpy as np

import glob
# coding : utf-8
import os
import abc
import xml.dom.minidom as xml


class_name = ["car", "van", "bus", "open_bed_heavy_truck", "close_bed_heavy_truck",
              "open_bed_light_truck", "close_bed_light_truck", "others_truck", "back_plate", "side_plate"]
# class_name = ["car", "van", "bus", "truck", "plate"]


class XmlReader(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    def read_content(self, filename):
        content = None
        if (False == os.path.exists(filename)):
            return content
        filehandle = None
        try:
            filehandle = open(filename, 'rb')
        except FileNotFoundError as e:
            print(e.strerror)
        try:
            content = filehandle.read()
        except IOError as e:
            print(e.strerror)
        if (None != filehandle):
            filehandle.close()
        if(None != content):
            return content.decode("utf-8", "ignore")
        return content

    @abc.abstractmethod
    def load(self, filename):
        pass
ROOT_PATH="/home/data"  # os.getcwd()


img_s_sum=[]
label_150=[]


class XmlTester(XmlReader):
    def __init__(self):
        XmlReader.__init__(self)

    def load(self, filename, use_color=False):
        filecontent=XmlReader.read_content(self, filename)
        if None != filecontent:
            # print(filename)
            dom=xml.parseString(filecontent)
            im_size=dom.getElementsByTagName('size')[0]

            im_w=int((im_size.getElementsByTagName(
                'width')[0]).childNodes[0].data)
            im_h=int((im_size.getElementsByTagName(
                "height")[0]).childNodes[0].data)
            self.im_shape=np.array([im_w, im_h])
            self.bbox=[]
            self.name=dom.getElementsByTagName(
                'filename')[0].childNodes[0].data
            self.obj_name=[]
            for obj in dom.getElementsByTagName('object'):
                box=obj.getElementsByTagName('bndbox')[0]

                b_xmin=int((box.getElementsByTagName(
                    "xmin")[0]).childNodes[0].data)
                b_ymin=int((box.getElementsByTagName(
                    "ymin")[0]).childNodes[0].data)
                b_xmax=int((box.getElementsByTagName(
                    "xmax")[0]).childNodes[0].data)
                b_ymax=int((box.getElementsByTagName(
                    "ymax")[0]).childNodes[0].data)
                b_name=obj.getElementsByTagName(
                    "name")[0].childNodes[0].data
                self.obj_name.append(b_name)
                # print(self.name)
                self.bbox.append([[b_xmin, b_ymin], [b_xmax, b_ymax]])
                if b_name == "open_bed_heavy_truck" and use_color:
                    attr1=obj.getElementsByTagName("attributes")[0]
                    # attr2 =attr1.getElementsByTagName("attribute")[1] #[0].childNodes[0].data

                    n_2=attr1.getElementsByTagName(
                        "value")[1].childNodes[0].data
                    if n_2.isdigit():
                        n_2=attr1.getElementsByTagName(
                            "value")[0].childNodes[0].data
                    img=cv2.imread("/home/data/1441/"+self.name)

                    img0=img[b_ymin:b_ymax, b_xmin:b_xmax]
                    if (img is None) or img0 is None:
                        print("?????????", self.name, img0)
                        continue
                    img_s=cv2.resize(img0, (150, 150))
                    img_s_sum.append(img_s)
                    label_150.append(n_2)

            self.polygon=[]
            # print(len(dom.getElementsByTagName('polygon')))
            for obj in dom.getElementsByTagName('polygon'):
                o_cls=obj.getElementsByTagName('class')[0].childNodes[0].data

                box=obj.getElementsByTagName(
                    'points')[0].childNodes[0].data.replace(";", ",").split(",")
                
                if class_name[-1] == str(o_cls) or class_name[-2] == str(o_cls):
                    self.obj_name.append(o_cls)
                    pts=np.array(box).astype(np.int32).reshape(-1, 2)
                    # pts=np.array().as
                    a=pts.min(0)
                    b=pts.max(0)
                    self.bbox.append([a, b])
                    continue
                self.polygon.append(
                    [o_cls, np.array(box).astype(np.int32).reshape(-1, 2)])
                    # print(self.bbox[-1])
            self.bbox=np.array(self.bbox)
            return self.im_shape, self.bbox

    def xyxy2yolo(self):
        self.bboxyolo=[]
        for bx in self.bbox:  # xyxy
            cent=(bx[1]+bx[0])/2
            dwh=bx[1]-bx[0]
            # self.bboxyolo.append([bx[0]/self.im_shape,dwh/self.im_shape])
            self.bboxyolo.append(
                [cent/self.im_shape, dwh/self.im_shape])  # cxcywh

        # for o_cls, pt in self.polygon:
        #     # print(o_cls[-5:])
        #     if o_cls[-5:]=="plate":
        #         # print(sorted(pt))
        #         plx=pt.min(0)
        #         prx=pt.max(0)
        #         self.bboxyolo.append([plx/self.im_shape,prx/self.im_shape])
        return np.array(self.bboxyolo).reshape(-1, 4), self.obj_name

    def imshow(self):
        img=cv2.imread(ROOT_PATH+"/"+self.name)
        for bx in self.bbox:
            cv2.rectangle(img, bx[0], bx[1], (0, 255, 0), 1, 1)
        # cv2.imshow("a", img)
        # cv2.waitKey()
        # img=cv2.cvtColor(img, cv2.COLOR_BGR2BGRA);
        for clas, point in self.polygon:
            # cv2.polylines(img, [point], 1, (0,0,0,255))
            # print(point)
            img=put_mask(img, point)
        size_decrease=(int(img.shape[1]/2), int(img.shape[0]/2))
        img_decrease=cv2.resize(
            img, size_decrease, interpolation=cv2.INTER_CUBIC)
        # mask = im
        # cv2.imshow("Mask", mask)
        # masked = cv2.bitwise_and(image, image, mask=mask)
        cv2.imshow("Mask to Image", img_decrease)
        cv2.waitKey(0)

def to_yolo_2d():

    xmls=glob.glob("%s/1441/*.xml" % ROOT_PATH)
    data=[]
    for i, xml_path in enumerate(xmls):
        reader=XmlTester()
        im_shape, bbox=reader.load(xml_path)
        flb=open(xml_path[:-3]+"txt", mode="w")
        b, ns=reader.xyxy2yolo()
        for b_, target in zip(b, ns):
            if class_name.count(target) != 0:
                indx=class_name.index(target)
                flb.write("%d %f %f %f %f\n" %
                          (indx, b_[0], b_[1], b_[2], b_[3]))
        flb.close()
        data.append([xml_path[:-3]+"jpg",im_shape,reader.polygon.copy()])
    import pickle

    pickle.dump(data,open("a.pkl","wb"))
    print("xmls:", len(xmls))
import datetime

start_time=datetime.datetime.now()  # ???????????????????????????????????????
to_yolo_2d()
stop_time=datetime.datetime.now()  # ?????????????????????????????????
func_time=stop_time-start_time  # ?????????????????????????????????
print("func is running %s s" % func_time)


import pickle
f=open("a.pkl",mode="rb")
md=pickle.load(f)
img=np.zeros((1080,1920),dtype=np.uint8)


class_map = {'dirt': 2, 'bg': 0, "truck_bed_cover": 1, "empty_bed": 3}

def point2img(idx):
    _,s,polgs=md[idx]
    image=np.zeros(s,dtype=np.uint8)
    for nm,pt in polgs:
        if len(pt):
            points = np.array(pt, dtype=np.int32)
            image=cv2.fillPoly(image, [points], color=(class_map[nm]*80,100,200))
    return image
for i in range(10):
    img=point2img(i)

    cv2.imshow("a",img)
    cv2.waitKey()
# 


# for nm,pg in (md[0][1]):
#     print(nm,pg)
#     points = np.array(pg, dtype=np.int32)
#     img=cv2.fillPoly(img, [points], color=(class_map[nm]*80))
# # print(img)
# cv2.imshow("a",img)
# cv2.waitKey()

