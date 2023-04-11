import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
#读取csv文件
# df = pd.read_csv('/home/liuyixuan/workspace/personalShuffle/RucPriv/utils/data/n50_005~03_epsacc2.csv')
df = pd.read_csv('/home/liuyixuan/workspace/personalShuffle/perS/utils/data/n1w_005_1_epsacc3.csv')
# df = pd.read_csv('/home/liuyixuan/workspace/personalShuffle/perS/utils/data/test.csv')
print(df.columns)
# rounds = df.loc[:, 'round']
color = ['-.','b-.', 'g-.', 'c-.',  'r', 'gold', 'orange']
plt.switch_backend('agg')

# for i in range(len(df.columns)):
#     if i==0:
#         continue
#     exp = df.columns[i]
#     y = df.loc[:,exp]
#     plt.plot(rounds, y, color[i])
p = []
# exp_name = ['Ours','PerLDP-min','PerLDP-max','Exist Shuffle','PerLDP','Ours+sample']
exp_name = ['LDP-min','LDP-max','LDP','UniS','PerS','PerSS','Ours+PostSpars']

ls = ['--', ':', '-.', '--', '-', '-', '-']
m = ['o', '+', '*', ',', '^', 's', '*']
c = ['slategrey', 'dodgerblue', 'blueviolet', 'darkcyan', 'yellowgreen', 'r', 'orange']
k=0
for i in range(0, len(df.columns), 2):
    if i<6 :
        continue
    # print(i,k)
    exp = df.columns[i]
    eps = df.loc[:,exp]
    i += 1
    acc_col = df.columns[i]
    acc = df.loc[:,acc_col]
    # plt.plot(acc, eps, label=exp, linestyle=ls[k], color=c[k], marker=m[k], markevery=10)
    plt.plot(eps, acc, label=exp, linestyle=ls[k], color=c[k], marker=m[k], markevery=10)
    k+=1

# 设置xy坐标范围
# plt.ylim((10**1,10**5))
# plt.xlim((.5,.8))
 
#xy描述
plt.ylabel('$\epsilon^C$')
plt.xlabel('Accuracy')
# plt.yscale('log')
# plt.ylabel('Accuracy')
# plt.xlabel('Rounds')
# print([exp for exp in exp_name])
plt.legend(loc='lower left', fontsize=8)
# plt.title('QMNIST Acuuracy v.s. Epsilon', fontsize='large', fontweight='bold')
# plt.title('MINISt Accuracy with the same DP 14.3', fontsize='large', fontweight='bold')
plt.title('Local privacy: Uniform2, $\delta_s=10^{-8}, n=10000, d=7850, C=0.1$', fontsize=9)
plt.show()
# plt.savefig('/home/liuyixuan/workspace/personalShuffle/perS/utils/01-1_epsacc2.png', dpi=600)# 设置dpi并保存图像到本地
plt.savefig('/home/liuyixuan/workspace/personalShuffle/perS/utils/005-1_epsacc3.png', dpi=600)
plt.close()