
import pandas as pd

# RULE1
dfa = pd.read_csv("data/report_clean.csv")
x = dfa.loc[dfa.InfoTitle.str.contains("财务决算|公开信息")]
a = dfa.index.to_list()
b = x[x.InfoTitle.str.contains("关于召开|提示性公告|法律意见书|议事规则|会议材料|会议资料|H股|召开通知")].index.to_list()
for i in b:
    a.remove(i)
xx = dfa.loc[a]
xx.to_csv("data/report_clean3.csv", encoding="utf-8-sig", index=None)


# # RULE2
# <code> for df.Title.str.contains("交易信息|公开信息"):<br/>&emsp; delete
# </code>
# Applied RULE2：df.shape(253001,16)->(249286,16)
#
#
# # RULE3
# <code> for df.infoTitle.str.contains("财务决算|年度审计报告|财务报告"):<br/>&emsp; delete
# </code>
# Applied RULE3：df.shape(249286,16)->(239248,16)
#
# # RULE4
# <code> for df.infoTitle.str.contains("董事会工作报告"):<br/>&emsp; delete
# </code>
# Applied RULE3：df.shape(239248,16)->(237211,16)
#
# # RULE5
# ? 述职报告