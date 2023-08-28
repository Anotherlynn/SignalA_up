import pandas as pd
import statsmodels.api as sm
from nltk.tbl import feature
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from math import isnan
import numpy as np
import matplotlib.colors as mcolors
from matplotlib import cm

plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

import random

import colorsys

# Load your data

def random_light_color():
    h = random.random()
    s = random.uniform(0.5, 0.8)
    v = random.uniform(0.9, 1.0)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return r, g, b


class psm():
    def __int__(self, data, tag, window=20):
        self.Tag = tag
        self.window = window
        # data['TradingDate'] = pd.to_datetime(data['TradingDate'])
        self.data = data.dropna(subset=['return'])
        self.colors = [random_light_color() for i in range(0, 150)]

    def add_vwap(self,df):
        '''
        # get vWap and vWap_adj(复权)
        # vWap = TurnoverValue/TurnOverVolume
        # vWap_adj = vWap/(PrevClosePrice/Close_t-1)
        :return:
        '''
        # df['vWap'] = df['TurnoverValue']/df['TurnoverVolume']
        df = df.sort_values(by='TradingDate').reset_index(drop=True)
        df['vWap_adj'] = df['vWap'] / (df['PrevClosePrice'] / df.shift(1)['ClosePrice'])
        df['return'] = df['vWap_adj'].pct_change()
        df = df.dropna(subset=['vWap', 'vWap_adj'], how='any').reset_index(drop=True)
        return df
    # indice_ = indice.groupby(['InnerCode'],as_index=False).apply(add_vwap)
    # window =30
    def get_var(self,dff):
        '''
        # get window days variation
        :return:
        '''
        # df = df.dropna(subset=['return_'],how='any')
        indice_ = dff.groupby(['InnerCode'], as_index=False).apply(self.add_vwap)
        dfff = indice_.sort_values(by='TradingDate').reset_index(drop=True)

        out_list = []

        for idx in range(0, dfff.shape[0]):
            sub_list = []
            sub_idx = max(0, int(idx - self.window / 2))
            end_idx = min(dfff.shape[0] - 1, int(idx + self.window / 2))
            # print("add from date: " + str(dfff['TradingDate'].values[sub_idx]) + " to date:  " + str(dfff['TradingDate'].values[end_idx]))

            for return_p in dfff['res'].values[sub_idx:end_idx + 1]:
                sub_list.append(return_p)

            out_list.append(sub_list)
        # out_list.append([df['TradingDate'].values[-1]])
        # print('idx [ ' + str(dfff['InnerCode'].unique()) + "] added!")
        dfff['res_'] = out_list
        return dfff



        # call by:
        # target_list =[]
        # df.groupby('InnerCode').apply(lambda x: target_list.append(get_var(x,window)))
        # target_list = sum(target_list,[])
        # df['return_indus'] = target_list
        #
        # # if mean is needed for certain categorical groups
        # b = df.groupby(['InnerCode','target_cat'],as_index=False)['return'].count()
        # b['return_mean'] = df.groupby(['target_cat','TradingDate']).apply(lambda x:get_var_mean(x,window)).values

    def get_var_mean(self, df_):

        out_list = []
        # out_len = []

        df = df_.sort_values(by='TradingDate').reset_index(drop=True)
        for return_ in df['return_'].values:
            # return_ = eval(return_.replace('nan' ,'np.nan'))
            out_list.append(return_)
            # out_len.append(len(return_))
        # print("Date:  " + str(df['TradingDate'][0:1]) + " Industry:  " + str(df['FstIndustryCSI'][0:1]))
        # 获取平铺后每个索引位置值在原始数列中出现的次数
        # counts = np.bincount(out_len)
        # # 返回众数 返回最大值在数列中的索引位置
        # len_ = np.argmax(counts)
        # for x in out_list[::-1]:
        #     if len(x)!=len_:
        #         out_list.remove(x)
        #         print('removed!')
        out_list_avg = []
        for i in range(0, self.window + 1):
            sum = 0
            count = 0
            for item in out_list:
                try:
                    item_val = item[i]
                    if isnan(item_val) == False:
                        sum += item_val
                        count += 1
                except:
                    sum += 0
                    count += 0
            if count != 0:
                out_list_avg.append(sum / count)
                # print('out_list_avg[%d] append!' % i)
            else:
                out_list_avg.append(np.nan)
                # print('out_list_avg[%d] failed append!' % i)

        return out_list_avg

    def get_diff(self,df):
        a = df.groupby('InnerCode', as_index=False).apply(self.get_var).reset_index(drop=True)
        b = a.groupby(['FstIndustryCSI','TradingDate'],as_index=False)['return'].count()
        b['return_avg'] = a.groupby(['FstIndustryCSI','TradingDate']).apply(lambda x: self.get_var_mean(x)).values
        df = pd.merge(a,b[['FstIndustryCSI','TradingDate','return_avg']], on=['FstIndustryCSI','TradingDate'], how='left')

        df = df.sort_values(by='TradingDate').reset_index(drop=True)
        df_list = []
        for idx, k in df.iterrows():
            s1 = k['return_']
            s2 = k['return_avg']
            # print(s1)
            out_list_avg = []
            for i in range(0, self.window + 1):
                try:
                    out_list_avg.append(s1[i] - s2[i])
                    # print('out_list_avg[%d] append!' % i)
                except:
                    out_list_avg.append(np.nan)
                    # print('out_list_avg[%d] failed append!' % i)
            df_out = pd.DataFrame(data=out_list_avg)
            df_out = df_out.fillna(method='ffill')
            df_list.append(df_out[0].values.tolist())
        df['return_diff'] = df_list
        return df

    def custome_fg(self,ax):
        # ax = plt.ravel()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_position(('data', 0))
        ax.set(xlabel='Time Window', ylabel='Price Change')
        ax.legend(loc='upper right')
        # ax.rcParams['font.sans-serif'] = ['SimHei']
        # # plt.rcParams['font.size'] = 16
        # ax.rcParams['axes.unicode_minus'] = False
        # plt.legend(loc='upper right')
        # plt.tight_layout()

    def plot_diff(self, savePath, sample_size=120):
        if len(self.Tag)==1:
            df = self.get_diff(self.data)
            plot_df = df[df[self.Tag[0]] == 1]

        else:
            plot_df = self.data

        plot_list1 = [i for i in plot_df['return_diff'].values if len(i) == self.window + 1]
        plot_list2 = [i for i in plot_df['return_diff_CSI'].values if len(i) == self.window + 1]

        mean_1 = np.array(list(np.mean(plot_list1, axis=0))) + 1
        mean_cum1 = np.hstack([1 / (mean_1[1:int(self.window / 2) + 1][::-1].cumprod()[::-1]), 1,
                               mean_1[int(self.window / 2) + 1:].cumprod()])
        mean_2 = np.array(list(np.mean(plot_list2, axis=0))) + 1
        mean_cum2 = np.hstack([1 / (mean_2[1:int(self.window / 2) + 1][::-1].cumprod()[::-1]), 1,
                               mean_2[int(self.window / 2) + 1:].cumprod()])

        x_ = list(range(-int(self.window / 2), int(self.window / 2) + 1))
        # color_map = plt.get_cmap('gnuplot2')  # Getting a colormap that can provide distinct colors
        if plot_df.shape[0]>=sample_size:
            df_sample = plot_df.sample(n=min(plot_df.shape[0], sample_size))

        else:
            df_sample = plot_df

        sample_plot_list1 = [i for i in df_sample['return_diff'].values if len(i) == self.window + 1]
        sample_plot_list2 = [i for i in df_sample['return_diff_CSI'].values if len(i) == self.window + 1]

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))
        # Plot data on each subplot

        axes[0, 0].plot(x_, mean_cum1, label=feature, color='b', linewidth=1.7)
        self.custome_fg(axes[0, 0])
        axes[0, 0].set_title(f'Effect Industry Avg.(Sample Size: {plot_df.shape[0]})')
        # axes[0, 0].set_xlabel('Time Window')
        # axes[0, 0].set_ylabel('Price Change')

        for i_ in range(0, len(sample_plot_list1)):
            mean_sample = np.array(sample_plot_list1[i_]) + 1
            mean_sample_cum = np.hstack([1 / (mean_sample[1:int(self.window / 2) + 1][::-1].cumprod()[::-1]), 1,
                                         mean_sample[int(self.window / 2) + 1:].cumprod()])
            print(i_)
            axes[1, 0].plot(x_, mean_sample_cum, color=self.colors[i_], alpha=0.6, linewidth=1.1, linestyle='--')

        axes[1, 0].plot(x_, mean_cum1, label=feature, color='b', linewidth=1.7)
        self.custome_fg(axes[1, 0])
        axes[1, 0].set_title(f'Effect Industry Avg.(subSample Size: {df_sample.shape[0]})')
        axes[1, 0].set_ylim(0.6, 1.5)

        axes[0, 1].plot(x_, mean_cum2, label=feature, color='b', linewidth=1.7)
        self.custome_fg(axes[0, 1])
        axes[0, 1].set_title(f'Effect Index CSI800.(Sample Size: {plot_df.shape[0]})')
        # axes[1, 0].set_xlabel('Time Window')
        # axes[1, 0].set_ylabel('Price Change')

        for ii_ in range(0, len(sample_plot_list2)):
            mean_sample2 = np.array(sample_plot_list2[ii_]) + 1
            mean_sample_cum2 = np.hstack([1 / (mean_sample2[1:int(self.window / 2) + 1][::-1].cumprod()[::-1]), 1,
                                          mean_sample2[int(self.window / 2) + 1:].cumprod()])
            axes[1, 1].plot(x_, mean_sample_cum2, color=self.colors[ii_], alpha=0.6, linewidth=1.1, linestyle='--')
        axes[1, 1].plot(x_, mean_cum2, label=feature, color='b', linewidth=1.7)
        self.custome_fg(axes[1, 1])
        axes[1, 1].set_title(f'Effect Index CSI800.(subSample Size: {df_sample.shape[0]})')
        axes[1, 1].set_ylim(0.6, 1.5)

        fig.tight_layout()
        # plt.rcParams['xtick.labelsize'] = 12
        # fig.show()
        # plt.rcParams['ytick.labelsize'] = 12
        # If you want to save the plot uncomment the below line
        plt.rcParams['font.sans-serif'] = ['SimHei']
        # plt.rcParams['font.size'] = 16
        plt.rcParams['axes.unicode_minus'] = False
        # fig.show()
        plt.savefig(savePath+'/PSM_20.png')
        plt.close()
        print('saved PSM!')
#
# ate_list = []
# for i in range(10):  # Assuming the return_ list has 10 values
#     treated_returns = treated['return_'].apply(lambda x: x[i])
#     matched_returns = matched_control['return_'].apply(lambda x: x[i])
#     ate = np.mean(treated_returns - matched_returns)
#     ate_list.append(ate)
#
# print(ate_list)

# def plot_diff(df,feature,window=20,sample_size=120):
#     plot_df = df[df[feature]==1]
#     plot_list = []
#     for i in plot_df['return_diff_index'].values:
#         if len(i)==window+1:
#             plot_list.append(i)
#
#     mean_ = np.array(list(np.mean(plot_list, axis=0)))+1
#     mean_cum = np.hstack([1/(np.array(mean_[:int(window/2)][::-1]).cumprod()[::-1]),1,np.array(mean_[int(window/2)+1:]).cumprod()])
#     # print(mean_cum)
#     x_ = [-10,-9,-8,-7,-6,-5, -4, -3, -2, -1, 0, 1, 2, 3, 4,5,6,7,8,9,10]
#     color = ['b','cornflowerblue','indigo','m','thistle','hotpink','slateblue','cyan','slategray','darkorange']
#     df_sample = plot_df.sample(n=min(plot_df.shape[0],sample_size))
#     size = df_sample.shape[0]
#     sample_plot_list = []
#     for i in df_sample['return_diff_index'].values:
#         if len(i)==window+1:
#             sample_plot_list.append(i)
#
#     plt.figure(figsize=(14, 8))
#     plt.plot(x_, mean_cum, label=feature, color='r',linewidth=1.5)
#     # plt.bar(x=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4], height=list(np.mean(plot_list, axis=0)), label='return_diff', color='Coral', alpha=0.8)
#
#     for i_ in range(0,size,30):
#         if i_+30<=sample_size:
#             sample_group = sample_plot_list[i_:i_+30]
#         else:
#             sample_group = sample_plot_list[i_:]
#         mean_sample = np.array(list(np.mean(sample_group, axis=0))) + 1
#         mean_sample_cum = np.hstack([1 / (np.array(mean_sample[:int(window / 2)][::-1]).cumprod()[::-1]), 1,
#                               np.array(mean_sample[int(window / 2) + 1:]).cumprod()])
#         plt.plot(x_, mean_sample_cum, label='sample_'+str(int(i_/30)), color=color[int(i_/30)-1],alpha=0.5,linewidth=1)
#     plt.xlabel('Datetime_window')
#     plt.ylabel('Price')
#     plt.title('Effect Over Time_sample_%d'%plot_df.shape[0])
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     plt.rcParams['font.size'] = 12
#     plt.rcParams['axes.unicode_minus'] = False
#     plt.rcParams['xtick.labelsize'] = 10
#     plt.rcParams['ytick.labelsize']= 10
#     # for aa, bb in zip([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4], a['return_diff']):
#     #     plt.text(aa, bb, bb, ha='center', va='bottom', fontsize=8)
#     ax = plt.gca()
#     ax.spines['right'].set_color('none')
#     ax.spines['top'].set_color('none')  # 将右边 上边的两条边颜色设置为空 其实就相当于抹掉这两条边
#     ax.xaxis.set_ticks_position('bottom')
#     ax.yaxis.set_ticks_position('left')  # 指定下边的边作为 x 轴   指定左边的边为 y 轴
#     # ax.spines['bottom'].set_position(('data', 0))  # 指定 data  设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上
#     ax.spines['left'].set_position(('data', 0))
#     plt.legend(loc='upper right')
#     plt.show()
#     # plt.savefig('result/PSM_20/%s_20.png'%feature)

# for feature in ['异常波动', '分红', '股东大会决议', '业绩预告',
#        '持股变动', '资产重组', '再融资', '股权激励', '关联交易', '担保', '退市风险', '交易所交易公开信息',
#        '现金管理', '会计政策变更', '人员聘请', '审计保留意见', 'IPO', '变更信息', '内部控制', '新项目开展',
#        '诉讼案件', '承诺澄清', '无法分类']:
#     plot_diff(df_use,feature,20)


# plt.switch_backend('agg')











# from fastparquet import ParquetFile
#
# datadir = r'data/'
# filename = datadir + r'factor_exposure.parq'
# pf = ParquetFile(filename)
# data = pf.to_pandas(index=False)
#
# outfile = datadir + r'data/factor_exposure.csv'
# data.to_csv(outfile, encoding='utf-8-sig')
#
# # 检验是否标准化
# import seaborn as sns
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# # 用来正常显示负号
# plt.rcParams['axes.unicode_minus'] = False
# plt.hist(data.年龄, bins=20)
#
#
# def wls_barra(df):
#     x_add = np.asarray(df[['BETA', 'RSTR', 'RVOL',
#                          'NLSIZE', 'BTOP', 'LIQUIDITY', 'EARNINGSYIELD', 'GROWTH', 'LEVERAGE',
#                          'AgroforestryFishing', 'Automobile', 'Bank', 'Building',
#                          'BuildingMaterials', 'Chemical', 'Coal', 'Communication',
#                          'Comprehensive', 'ComprehensiveFinance', 'Computer',
#                          'ElectronicComponents', 'FoodBeverage', 'HouseholdHousehold',
#                          'Manufacture', 'Mechanics', 'Media', 'Medicine', 'Military',
#                          'NonBankFinance', 'NonferrousMetal', 'Petrochemical', 'PowerEquipment',
#                          'RealEstate', 'RestaurantTour', 'Retail', 'Steel', 'TextileClothing',
#                          'Transportation','Utility', 'COUNTRY']])
#     y = np.asarray(df['return'])
#     model = sm.WLS(y,x_add,hasconst=True,weights=np.sqrt(df['SIZE_ln'])).fit()
#     y_pred = model.predict(x_add)
#     # w = np.diag(1/np.sqrt(df['SIZE_ln']))
#     # w_inv = np.linalg.inv(w)
#     #
#     # # 约束矩阵R
#     # # k = x_add.shape[1]
#     # # diag_R = np.diag(np.ones(k))
#     # # location = len
#     # mat = np.linalg.inv(np.dot(np.dot(np.transpose(x_add), w_inv), x_add))
#     # param = np.dot(np.dot(np.dot(mat, np.transpose(x_add)), w_inv), y)
#     # y_pred = np.dot(x_add, param)
#     # # Residuals
#     residuals = y - y_pred
#     df['return_pred'] = y_pred
#     df['res'] = residuals
#     return df
#
# def sort_tick(df):
#     df = df.sort_values(by='datetime').reset_index(drop=True)
#     df_new = df.shift(1)
#     df_new['TradingDate'] = df['datetime'].apply(lambda x:str(x).split()[0])
#     df_new['ticker'] = df['ticker']
#     df_new = df_new.drop(['datetime'],axis=1)
#     print('ok')
#     return df_new
# from sklearn.preprocessing import StandardScaler
# b_test=StandardScaler()      #训练数据，赋值给b_test
# X_result=b_test.fit_transform(X_test)
#
# # fac_date 因子时间
# # sig-date 交易时间
# # symbol = ticker
