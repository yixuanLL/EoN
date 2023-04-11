import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
#读取csv文件
df = pd.read_csv('/home/liuyixuan/workspace/personalShuffle/perS/utils/data/n1w_005_1_topk.csv')

print(df.columns)
rounds = df.loc[:, 'round']
ls = ['--', '--', '--', '--', '-', '-', '-','-']
m = ['*', 'v', 'x', '', '*', 'v', 'x','']
c = ['yellowgreen', 'r', 'orange','dodgerblue', 'yellowgreen', 'r', 'orange','dodgerblue']
plt.switch_backend('agg')

for i in range(len(df.columns)):
    if i==0:
        continue
    exp = df.columns[i]
    y = df.loc[:,exp]
    if i == 7:
        plt.plot(rounds, y, label=exp, linestyle=ls[i-1], color=c[i-1], marker=m[i-1], markevery=8)
    else:
        plt.plot(rounds, y, label=exp, linestyle=ls[i-1], color=c[i-1], marker=m[i-1], markevery=5)


# 设置xy坐标范围
# plt.ylim((10**2,10**5))
# plt.xlim((.45,.8))
 
#xy描述
plt.xlabel('round')
plt.ylabel('accuracy')
# plt.yscale('log')
# plt.ylabel('Accuracy')
# plt.xlabel('Rounds')
# print([exp for exp in exp_name])
plt.legend(loc='lower right', fontsize=9)
plt.title('Local privacy: Uniform2, $\delta_s=10^{-8}, n=10000, d=7850, C=0.1$', fontsize=9)
# plt.title('MINISt Accuracy with the same DP 14.3', fontsize='large', fontweight='bold')
plt.show()
# plt.savefig('/home/liuyixuan/workspace/personalShuffle/perS/utils/01-1_epsacc2.png', dpi=600)# 设置dpi并保存图像到本地
plt.savefig('/home/liuyixuan/workspace/personalShuffle/perS/utils/005-1_topk.png', dpi=600)
plt.close()