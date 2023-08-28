#!./.conda/envs/py310/bin/python3.10

# This is the main file for SignalA_up project
# author: Xinyi Li
# contact: xl4412@nyu.edu
# time of completion: 2023-08-10
import pandas as pd
import exchange_calendars as trade_date

from proj.mergedata import Preprocessor
# from topic_model import LDA
from proj.output import event_tag
import warnings
import numpy as np
warnings.filterwarnings("ignore")


## generate new trading time
#
# import jieba
#
# # names
# jieba.load_userdict("data/organization.txt")
# jieba.load_userdict("data/organization2.txt")
# jieba.load_userdict("data/person.txt")
# # cities
# jieba.load_userdict("data/glossary.txt")


if __name__ == "__main__":


    data = pd.read_csv('data/sample.csv',index_col=0)
    save_file = input("请输入想要输出文件的绝对文档位置，格式：mac/doc，按回车输入 ：\n")
    Tag = input("请输入要分析的事件，用逗号分隔开。\n\n可选择的事件有：异常波动, 分红, 股东大会决议, 业绩预告,\n持股变动, 资产重组, 再融资, 股权激励,\n关联交易, 担保, 退市风险, 交易所交易公开信息,\n"
                "现金管理, 会计政策变更, 人员聘请, 审计保留意见,\nIPO, 变更信息, 内部控制, 新项目开展, 诉讼案件, 承诺澄清\n")

    # save_file = "./result"
    Tag = Tag.replace("，",",")
    Tag = Tag.replace(" ",",")
    a = Tag.split(",")
    b = [i!='' for i in Tag.split(",")]
    Tag_ = []
    for i, t in zip(a,b):
        if t:
            Tag_.append(i)
    report1 = event_tag(data,Tag=Tag_, save_file = save_file)
    report1.reports()


    # calendar = trade_date.get_calendar("XSHG")
    #
    # # load your report data
    # dataPath1 = './'
    # data_report = pd.read_csv(dataPath1)
    #
    # # load your price data
    # dataPath2 = './'
    # data_price = pd.read_csv(dataPath2)
    #
    # # before you go, use tools in Preprocessor to clean your data
    #
    # pre = Preprocessor(data_report)
    #
    # # if you want to check if there is any duplicated of Company:
    # pre.check_duplicate()
    #
    # # get real time of reports
    # data_ = pre.get_Ttime()
    #
    # # Name Entity Recognition(NER) to extract organizations, person name and place name
    # org, person, place = pre.get_entity('zh_core_web_sm')
    #
    # # save result
    # with open("SignalA_up/data/organization.txt", 'w') as f:
    #     for i in org:
    #         f.write(str(i) + '\n')
    # f.close()
    #
    # with open("SignalA_up/data/person.txt", 'w') as f:
    #     for i in person:
    #         f.write(str(i) + '\n')
    # f.close()
    #
    # with open("SignalA_up/data/place.txt", 'w') as f:
    #     for i in place:
    #         f.write(str(i) + '\n')
    # f.close()
    #
    # # merge your cleaned report data and the price data, make sure they align
    #
    # data_combined = pre._mergedf(data_price, data_)
    #
    #
    # # Topic Modeling to get an estimate of Tags
    # topic_Model1 = LDA(data_,loadPath = './text_load_path',stopword_path='./your_stop_word_path')
    #
    # # save your dictionary for once, and the function will automatically load them for the following
    # topic_Model1.save_word()
    #
    # # passes means how many times to train the model
    # model = topic_Model1.LDA_model(num_topics=10,passes=60)
    #
    # # several functions are included for analysis, and you can also analyze using the model defined above
    # # or even analyze using your own model by loading it
    # # make sure your model is an LDA model saved by gensim
    #
    # # topic words visualization
    # topic_Model1.topic_visualization(model)
    #
    # # word count
    # print(topic_Model1.word_count())
    #
    # # sentense visualization (topic+source sentence)
    # # start and end define how many words you want to print out
    # topic_Model1.sentences_chart(start=0,end=20)
    #
    # # save your model
    # topic_Model1.save_model(savePath='./your_save_path')


