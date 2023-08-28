#!./.conda/envs/py310/bin/python3.10

# This is the tool file for data preprocessing
# author: Xinyi Li
# contact: xl4412@nyu.edu
# time of completion: 2023-07-20

import pandas as pd
import datetime
import exchange_calendars as trade_date
import re
import spacy  # 3.5.0


## Using the trading calender of Shanghai Stock Exchange
calendar = trade_date.get_calendar("XSHG")

class Preprocessor():
    def __int__(self,df):
        self.df = df

    ## Generate new trading time, the time for the announcement to be used

    # Notice that all the date data downloaded from the SQL JYDB are in the format of "%Y-%m-%d %H:%M:%S.%f"
    # for example: "2023-02-03 17:20:12.020"

    # Logic here is:
    # 如果入库时间比发布时间晚1天，那么TradingDate按较晚的入库时间录入
    # 如果入库时间和发布时间在同一天，那么TradingDate按入库时间录入
    # 如果入库时间比发布时间早1天，那么TradingDate按发布时间录入
    # [已检查]不存在周一发布，大于一天前录入的情况
    # [已检查]不存在周五发布，周日录入的情况
    # [已检查]不存在BulletinDate在周六周日的数据，RecordDate不在当天的情况

    def get_Ttime(self):
        # Initiate blank list to store date
        TradingTime = []
        Htime = []
        # formatting datetime
        self.df['BulletinDate'] = pd.to_datetime(self.df['BulletinDate'])
        self.df['InsertTime'] = pd.to_datetime(self.df['InsertTime'])

        self.df['diff'] = self.df['InsertTime']-self.df['BulletinDate']
        self.df['daydiff'] = [i.days for i in self.df['diff']]

        self.df = self.df.loc[[i[0] for i in self.df.iterrows() if (i[1]['daydiff'] in [-1,0,1])]]
        for id, i in self.df.iterrows():
            if i['daydiff'] not in [-1, 0, 1]:
                time = None
                ttime = None
            else:
                if i['daydiff'] == 1:
                    t = datetime.datetime.strftime(str(i['InsertTime'])[:20], "%Y-%m-%d %H:%M:%S")
                    if t.weekday() == 5 or t.weekday() == 6:
                        time = calendar.date_to_session(t, direction="next").strftime("%Y-%m-%d")
                    else:
                        time = datetime.datetime.strftime(i['InsertTime'], "%Y-%m-%d")
                    ttime = datetime.datetime.strftime(i['InsertTime'], "%H:%M:%S")

                else:
                    if i['daydiff']==-1:
                        if i['BulletinDate'].weekday() == 5 or i['BulletinDate'].weekday() == 6:
                            time = calendar.date_to_session(i['BulletinDate'], direction="next").strftime("%Y-%m-%d")
                        else:
                            time = datetime.datetime.strftime(i['BulletinDate'], "%Y-%m-%d")
                        ttime = "00:00:00"
                        # ttime = None
                    elif i['daydiff']==0:
                        if i['BulletinDate'].weekday() == 5 or i['BulletinDate'].weekday() == 6:
                            time = calendar.date_to_session(i['BulletinDate'], direction="next").strftime("%Y-%m-%d")
                        else:
                            time = datetime.datetime.strftime(i['BulletinDate'], "%Y-%m-%d")
                        ttime = datetime.datetime.strftime(i['InsertTime'], "%H:%M:%S")

            TradingTime.append(time)
            Htime.append(ttime)
            print('%d Success'%id)
        self.df['TradingDate'] = TradingTime
        self.df['TradingTime'] = Htime
        print("公告数据清洗完成")
        return self.df

    # Name Entity Recognition
    def get_entity(self,model='zh_core_web_sm'):
        nlp = spacy.load(model)
        # 合并标题与内容时，用\n连接标题与内容，以及内容的项目
        text_ = "\n".join(["\n".join([i[1]['InfoTitle'],i[1]['Detail']]) for i in self.df.iterrows()])
        doc = nlp(text_)

        org = []
        person = []
        place = []
        for ent in doc.ents:
            res = re.findall('[\u4e00-\u9fa5]', ent.text)
            txt = "".join(res)
            if ent.label_ == 'ORG' and txt not in org and len(txt)>1:
                org.append(txt)
            if ent.label_ == 'PERSON' and txt not in person and len(txt) > 1:
                person.append(txt)
            if (ent.label_ == 'GPE' or ent.label_=='LOC') and txt not in place and len(txt) > 1:
                place.append(txt)

        print("实体识别完成")
        return org,person,place

    def _mergedf(df1, df2):
        '''
        :param df1: 行情数据
        :param df2: 公告
        :return: combined df
        '''
        GroupList = []
        groups = df1.groupby('InnerCode')
        for g in groups:
            frame_ = df2[df2['InnerCode'] == g[0]]
            infolist = []
            for i in range(0, g[1].shape[0]):
                if i == 0:
                    info = None
                else:
                    sDate = g[1].loc[i - 1, 'TradingDay'].split()[0]
                    eDate = g[1].loc[i, 'TradingDay'].split()[0]
                    # 看一下是不是要判断none
                    f = frame_[sDate < frame_['TradingDate']]
                    f = f[f['TradingDate'] <= eDate]
                    if f.empty:
                        print("!")
                        info = None
                    else:
                        info = "\n\n".join([inf for inf in f['Detail']])
                infolist.append(info)
            g[1]['Info'] = infolist
            print("%s加载完成！" % str(g[0]))
            GroupList.append(g[1])
        result = pd.concat(GroupList)
        return result

    def check_duplicate(self):
        '''
        check if ther is any duplicate companycode reagrading one InnerCode
        :return:
        '''
        group = self.df.groupby(['InnerCode'], as_index=False)
        for g in group:
            len = g[1]['CompanyCode'].nunique()
            if len != 1:
                print("%s有非重复值！！" % str(g[0][0]))
                for i in g[1]['CompanyCode'].unique():
                    print(str(i) + '\n')
            else:
                print("%s无重复值" % str(g[0][0]))



##############################################################

# data['InfoType'].unique()
# 信息类别(InfoType)与(CT_SystemConst)表中的DM字段关联，令LB = 1311，得到信息类别的具体描述：
# 10-发行上市书，20-定期报告，30-业绩快报，50-章程制度，60-更正公告，70-临时公告，90-交易所通报，91-交易所临时停(复)牌公告，99-其他
# 目前发现有30和70
# 缺少的：# 100201 业绩预增？？？ 100201-业绩预增m100203-业绩预减，100204-业绩预盈，100205-扭亏为盈，100206-由盈转亏, 100701-特别处理公告？？？？

# 公告二级分类(CategoryLevel2)与(CT_SystemConst)表中的DM字段关联，令LB = 1734，得到公告二级分类的具体描述：
# 100102-第三季度报告全文, 100105-定期报告摘要更正补充公告，100106-定期报告更正补充公告，100107-利润分配公告
# 100108-年度报告，100109-中期报告，100202-业绩预亏，100207-其他业绩预测报告，100301-业绩快报，100406-发行公告，100408-受限股上市公告，100410-上市公告书
# 100603-配股获准公告，100604-配股提示公告，100702-暂停上市公告，100704-终止上市，100801-持股变动公告，100802-股权收购公告，100806-股权换购公告，100809-公司减资、分立事项
# 100901-资产重组公告，100902-重大合同公告，100903-借贷事项公告，100904-担保事项公告，100905-诉讼仲裁公告，100906-项目投资公告，100907-委托理财公告，100908-政策优惠公告
# 100909-关联交易公告，100910-股权激励，100912-募集资金情况，100913-其他融资事项，100914-处罚整改公告，100915-停牌公告，101001-高管变动公告，101002-公司制度文件
# 101101-董事会决议，101102-监事会决议，101103-股东大会召开公告，101201-补充更正公告，101202-基本情况变更，101204-股价异动，101205-风险提示，101206-临时公告，200101-债券发行公告，200105-可转换债券上市公告书
# 200202-付息公告，200203-兑付公告，200301-跟踪评级报告，700801-其他公告，800101-中国证监会，800201-上交所，800301-深交所，800402-法律意见书，800601-保荐意见书
#####################################################################################################