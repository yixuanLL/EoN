import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from calibration import Calibration

# n_list = range(1000, 10000, 100)
y_th_list = []
y_exp_list = []
for n in range(1000):
    c = Calibration()
    y_th, y_exp = c.Error(10000,'MixGauss2')
    y_th_list.append(y_th)
    y_exp_list.append(y_exp)
plt.switch_backend('agg')
plt.hist(y_th_list, bins=50, label='trials', facecolor='orange', edgecolor='dodgerblue', alpha=0.8)
# plt.plot(n_list, y_exp, label='theory eps=Uniform2', linestyle='-', color='yellowgreen', marker='*', markevery=10)

# 设置xy坐标范围
# plt.ylim((10**1,10**5))
# plt.xlim((-0.01,0.01))
 
#xy描述
plt.ylabel('Frequence')
plt.xlabel('Error')
# plt.yscale('log')

plt.legend(loc='upper right', fontsize=9)

plt.title('Local privacy: Uniform2, n=10000, C=0.1, A=0.1, Number of trials=10000', fontsize=9)
plt.show()
plt.savefig('/home/liuyixuan/workspace/personalShuffle/perS/utils/calibration_err_other_data_dist.png', dpi=600)
plt.close()

