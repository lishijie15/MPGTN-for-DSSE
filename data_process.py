import pandas as pd
import math
import numpy as np

def strfill(src, lg, str1):
    # math.ceil是向上取整，这个下面会具体介绍
    n = math.ceil((lg - len(src)) / len(str1))
    newstr = src + str1 * n
    return newstr[0:lg]


# YS_out = pd.read_csv('./transfer/MegaCRN_YS_out.csv', header=None)
# v = YS_out.values
# v = v[:, 5]
# v = np.around(v, 2)
#
# with open('D:/BaiduSyncdisk/李世杰_博士工作整理/联合预测建模/Power_traffic_coordinated/8500node/test/Loads.dss', 'r') as f:
#     str_temp = f.read().split("\n")[:-1]
# head_str = str_temp[:12]
# str_temp = str_temp[12:1189]
# str(v[0])
# type(strfill(str(v[0]), 7, " "))
#
# with open('D:/BaiduSyncdisk/李世杰_博士工作整理/联合预测建模/Power_traffic_coordinated/8500node/test/Loads_test.dss', 'w+') as f:
#     for str_ in head_str:
#         f.write(str_ + "\n")
#     for i in range(len(str_temp)):
#         str_ = str_temp[i]
#         v_ = v[i]
#         # print(strfill(str(v_), 7, " "))
#         str_ = str_[:98] + strfill(str(v_), 7, " ") + str_[98 + 7:]
#         f.write(str_ + "\n")

def PowerFlow_transition(pf_mode=None):  # args.pf_mode
    YS_out = pd.read_csv('D:/BaiduSyncdisk/李世杰_博士工作整理/联合预测建模/Power_traffic_coordinated/8500node/test/MegaCRN_YS_out.csv',
                     header=None)
    v = YS_out.values
    v = v[:, 5]
    v = np.around(v, 2)

    with open('D:/BaiduSyncdisk/李世杰_博士工作整理/联合预测建模/Power_traffic_coordinated/8500node/test/Loads.dss', 'r') as f:
        str_temp = f.read().split("\n")[:-1]
    head_str = str_temp[:12]
    str_temp = str_temp[12:1189]
    str(v[0])
    type(strfill(str(v[0]), 7, " "))

    with open('D:/BaiduSyncdisk/李世杰_博士工作整理/联合预测建模/Power_traffic_coordinated/8500node/test/Loads_test.dss', 'w+') as f:
        for str_ in head_str:
            f.write(str_ + "\n")
        for i in range(len(str_temp)):
            str_ = str_temp[i]
            v_ = v[i]
            str_ = str_[:98] + strfill(str(v_), 7, " ") + str_[98 + 7:]
            f.write(str_ + "\n")