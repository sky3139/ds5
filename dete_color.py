from PIL import Image as im
import os
import numpy as np
import cv2
from statistics import mode
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pickle
import cv2
import numpy as np

#coding : utf-8
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


ROOT_PATH = "/home/data"  # os.getcwd()


class XmlTester(XmlReader):

    def __init__(self):
        XmlReader.__init__(self)

    def load(self, filename, use_color=False):

        img_s_sum = []
        label_150 = []

        filecontent = XmlReader.read_content(self, filename)
        if None != filecontent:
            dom = xml.parseString(filecontent)
            im_size = dom.getElementsByTagName('size')[0]
            im_w = int((im_size.getElementsByTagName(
                'width')[0]).childNodes[0].data)
            im_h = int((im_size.getElementsByTagName(
                "height")[0]).childNodes[0].data)
            self.im_shape = np.array([im_w, im_h])
            self.bbox = []
            self.name = dom.getElementsByTagName(
                'filename')[0].childNodes[0].data
            self.obj_name = []
            for obj in dom.getElementsByTagName('object'):
                box = obj.getElementsByTagName('bndbox')[0]

                b_xmin = int((box.getElementsByTagName(
                    "xmin")[0]).childNodes[0].data)
                b_ymin = int((box.getElementsByTagName(
                    "ymin")[0]).childNodes[0].data)
                b_xmax = int((box.getElementsByTagName(
                    "xmax")[0]).childNodes[0].data)
                b_ymax = int((box.getElementsByTagName(
                    "ymax")[0]).childNodes[0].data)
                b_name = obj.getElementsByTagName(
                    "name")[0].childNodes[0].data
                self.obj_name.append(b_name)
                # print(self.name)
                self.bbox.append([[b_xmin, b_ymin], [b_xmax, b_ymax]])
                if b_name == "open_bed_heavy_truck" and use_color:
                    attr1 = obj.getElementsByTagName("attributes")[0]
                    # attr2 =attr1.getElementsByTagName("attribute")[1] #[0].childNodes[0].data

                    n_2 = attr1.getElementsByTagName(
                        "value")[1].childNodes[0].data
                    if n_2.isdigit():
                        n_2 = attr1.getElementsByTagName(
                            "value")[0].childNodes[0].data
                    img = im.open(filename[:-3]+"jpg")
                    # print("", self.name)
                    if (img is None):
                        print("空图片", self.name)
                        continue
                    # img0 = img[b_ymin:b_ymax, b_xmin:b_xmax]
                    img0 = img.crop((b_xmin, b_ymin, b_xmax, b_ymax))
                    if img0 is None:
                        print("空图片", )
                        continue
                    img_s = img0.resize((150, 150))
                    img_s_sum.append(np.array(img_s))
                    label_150.append(n_2)
            return img_s_sum, label_150


# start_time = datetime.datetime.now()  # 记录程序开始执行的当前时间
# to_color_2d()
# stop_time = datetime.datetime.now()  # 记录执行结束的当前时间
# func_time = stop_time-start_time  # 得到中间功能的运行时间
# print("func is running %s s" % func_time)


class_map = {"unknown": 4, 'red': 2, 'blue': 0, "green": 1, "yellow": 3}
CLASS_NUM = 5


class Net(nn.Module):

    def __init__(self, class_num=4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(6272, 512), nn.ReLU(),
            nn.Linear(512, class_num), nn.ReLU())
        # self.initialize_weights()

    def forward(self, x):
        return self.layers(x)

    def view(self):
        # print(model)
        x = torch.rand(size=(1, 3, 150, 150), dtype=torch.float32)
        for layer in self.layers:
            x = layer(x)
            print(layer.__class__.__name__, 'output:\t', x.shape)

    def initialize_weights(self):
        for m in self.layers():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


def toTensor(img):
    img = torch.from_numpy(img.transpose((0, 3, 1, 2)))
    return img.float().div(255).cuda()  # .unsqueeze(0)


ROOT_PH = os.getcwd()  # "/Users/kregan/School/ML/final/"


class LoadImages_pkl:
    def get_las(self):

        for imgp in self.imgs:
            xmlp = imgp[:-3]+"xml"
            r = XmlTester()
            read = r.load(xmlp, True)

    def __init__(self, path, img_size=150, stride=32, auto=True):

        data = pickle.load(open("data.pkl", "rb"))
        self.imgs = data['img']
        self.lbs = data["label"]
        self.nf = len(self.imgs)  # number of files
        self.rand = np.arange(0, self.nf)
        np.random.seed()
        # np.random.choice(X, 3, replace=False)

    def __iter__(self):
        
        np.random.shuffle(self.rand)
        # cv2.imwrite("t.jpg", self.imgs[self.rand[0]])
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf-8:
            raise StopIteration
        self.count += 1
        imgs = []
        lbs = np.zeros((3, CLASS_NUM))
        for i in range(3):
            pid = self.rand[self.count+i]
            img0 = self.imgs[pid]
            imgs.append(img0)
            # print(class_map[self.lbs[pid]])
            lbs[i, class_map[self.lbs[pid]]] = 1
        # print(lbs)
        img_d = toTensor(np.array(imgs))
        lb = torch.cuda.FloatTensor(lbs)
        return img_d, lb

    def __len__(self):
        return self.nf  # number of files


class cy():
    def __init__(self, weight=None):
        # if weight==None else torch.load(weight)

        if weight == None or not os.path.exists(weight):
            self.cnnmodel = Net(CLASS_NUM)
        else:
            self.cnnmodel = torch.load(weight)
        self.cnnmodel = self.cnnmodel.cuda()
        self.cnnmodel.eval()
        # for m in self.cnnmodel.modules():
        #     if isinstance(m, (nn.Conv2d, nn.Linear)):
        #         nn.init.xavier_uniform_(m.weight)
        self.save_name="/project/train/models/color_m.pt"
        self.criterion = nn.CrossEntropyLoss()  # 交叉熵，标签应该是0,1,2,3...的形式而不是独热的
        self.optimizer = optim.SGD(
            self.cnnmodel.parameters(), lr=0.0005, momentum=0.9)
        self.dataloader = LoadImages_pkl(8)
        self.epoch=0

    def train(self, data=None):
        min_los = 1000
        for each_epoch in range(20):
            self.epoch+=1
            running_loss = 0.0
            self.cnnmodel.train()
            if data == None:
                data = self.dataloader
            for index, (inputs, labels) in enumerate(data):
                outputs = self.cnnmodel(inputs)
                # print(inputs, labels)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()  # 清理上一次循环的梯度
                loss.backward()  # 反向传播
                self.optimizer.step()  # 更新参数
                running_loss += loss.item()
                loss.bactorching_loss = 0.0
                if index%30==0:
                    print(".",end='')
            # 保存每一轮的模型
            now_loss = running_loss
            if now_loss < min_los and each_epoch % 10 == 9:
                min_los = now_loss
                self.save()
            print("[{}] loss: {:.4f}".format(each_epoch, now_loss))

    def train_epo(self, data=None):
        min_los = 1
        running_loss = 0.0

        inputs, labels = data
        outputs = self.cnnmodel(inputs)
        # print(inputs, labels)
        loss = self.criterion(outputs, labels)
        self.optimizer.zero_grad()  # 清理上一次循环的梯度
        loss.backward()  # 反向传播
        self.optimizer.step()  # 更新参数
        running_loss += loss.item()
        loss.bactorching_loss = 0.0
        return running_loss
        # # 保存每一轮的模型
        # now_loss = running_loss/100
        # if now_loss < min_los and each_epoch % 10 == 9:
        #     min_los = now_loss
        #     torch.save(self.cnnmodel,
        #                 "/project/train/models/color.pt")  # 保存全部模型
        #     print("saved")
        # print("[{}] loss: {:.4f}".format(each_epoch, now_loss))

    @torch.no_grad()
    def pre(self, path):
        img0 = cv2.imread(path)  # BGR
        img0 = cv2.resize(img0, (150, 150))
        inputs = toTensor(np.array([img0]))

        outputs = self.cnnmodel(inputs)
        print(outputs.cpu()[0].numpy())

    def save(self):
        state = {"state_dict":self.cnnmodel.state_dict(),"class_map":class_map, "epoch":self.epoch}
        torch.save(state,self.save_name)  # 保存全部模型
        print("saved:",self.save_name)
    def load(self,dir):
        checkpoint = torch.load("test.pt")
        self.cnnmodel.load_state_dict(checkpoint["state_dict"])
        self.epoch = checkpoint["epoch"] + 1
#

def train_test():
    # ccc=cy(weight="/project/train/models/color.pt")
    ccc=cy()
    ccc.save_name="color_m.pt"
    # ccc.save("/project/train/models/color1.pt")
    ccc.train()
    # ccc.save()
train_test()