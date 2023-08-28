import pandas as pd
from proj.build_model import XGB_clf
from sklearn.metrics import fbeta_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import os
from PIL import Image

from proj.PSM import psm

import warnings
warnings.filterwarnings("ignore")


def mkdir(save_file):
    folder = os.path.exists(save_file)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(save_file)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print
        "---  new folder...  ---"
        print
        "---  OK  ---"
    return save_file

class event_tag():
    def __init__(self, data, Tag, save_file):
        self.Tag = Tag
        self.save_file = save_file
        save_file_ = save_file+"/"+self.Tag[0]
        self.savePath = mkdir(save_file = save_file_)
        data['tag_trans'] = data['tag_trans'].apply(eval)
        data['embedding'] = data['embedding'].apply(eval)
        self.data = data


    def reports(self):
        '''
        export reports
        '''

        # if type(self.Tag) == type('异常波动'):
        #     self.Tag = self.tag_to_list(self.Tag)

        print('%s:事件统计报告' %",".join([str(i) for i in self.Tag]))
        report_data = self.data[self.data.apply(lambda x: True==all([x[i]==1 for i in self.Tag]),axis=1)].reset_index(drop=True)

        if report_data.shape[0]!= 0:
            report_data.to_excel(self.savePath+"/样本:"+str(report_data.shape[0])+".xlsx")
            print("事件样本已保存！")

            # if len(self.Tag) == 1:
            clf = XGB_clf(if_save=True)
            # build model
            clf.bi_clf(self.data, self.Tag, savePath=self.savePath)
            clf.plot_result(self.savePath)
            clf.ala_result('x')

            try:
                psm_ = psm(self.data, self.Tag)
                psm_.plot_diff(self.savePath,120)
                print("complete PSM")

            except:
                print("failed PSM")
                disk = './result/PSM_barra_20_3/'
                for i in range(0,len(self.Tag)):
                    img_path = disk+str(self.Tag[i])+"_20.png"
                    img = Image.open(img_path)  # 读取图片
                    img.save(self.savePath+"/"+str(self.Tag[i])+".png")  # 保存图片

            # else:
            #     clf = XGB_clf(if_save=True)
            #     # build model
            #     clf.bi_clf(self.data, self.Tag, savePath=self.savePath)
            #     clf.plot_result(self.savePath)
                # clf.ala_result('x')

                # try:
                #     psm_ = psm(report_data, self.Tag)
                #     psm_.plot_diff(self.savePath, 50)
                #     print("complete PSM")
                # # clf = XGB_clf(if_save=True)
                # # # build model
                # # clf.bi_clf(self.data, self.Tag[0], savePath=self.savePath)
                # # clf.plot_result(self.savePath)
                # # clf.ala_result('x')
                # except:
                #     print("failed PSM")
                #     disk = './result/PSM_barra_20_3/'
                #     for i in self.Tag:
                #         img_path = disk + str(i) + "_20.png"
                #         img = Image.open(img_path)  # 读取图片
                #         img.save(self.savePath+'/'+str(i)+'_20.png')  # 保存图片

        else:

            print('没有共同发生这些事情的样本！将展示可能的组合事件的分析；若不存在任何事件组合，则单独展示事件分析！')

            # for i in range(0,len(self.Tag)):
            #     if os.path.exists(self.save_file+"/"+self.Tag[i]):
            #         print("事件%s分析已存在"%self.Tag[i])
            #     else:
            #         rr = event_tag(self.data, Tag=self.Tag[i], save_file=self.save_file+"/"+self.Tag[i])
            #         rr.reports()




