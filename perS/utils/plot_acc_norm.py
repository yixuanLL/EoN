import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
 
#读取csv文件
# # df = pd.read_csv('/home/liuyixuan/workspace/personalShuffle/RucPriv/utils/data/n50_005~015_C.csv')
# # print(df.columns)
# rounds = df.loc[:, 'round']
color = ['r','r', 'b', 'g-.', 'c-.', 'k-.', 'm-.', 'c-.']
plt.switch_backend('agg')

acc_perS = [0.72091002,0.77597137,0.796728016,0.777096115,0.74309816,0.67694274,0.631083845]
acc_uniS = [0.756799591,0.785582822,0.775408998,0.666768916,0.549182004,0.398619632,0.332719836]
norm_list = [0.01,0.05,0.1,0.3,0.5,0.8,1]
x = list(range(len(norm_list)))
w = 0.3


plt.bar(x, acc_perS,color='orange', width=w,label='CLap') #FF6347
# plt.bar(x, acc_perS,color='moccasin', width=w,label='CLap') #FF6347
x1=np.array(x)
print(x1)
for i in range(len(x)):
    x[i] += w
plt.bar(x, acc_uniS,color='yellowgreen', width=w,label='Lap',tick_label = norm_list) #FFE4E1
# for i in range(len(x)):
#     x[i] += w
# plt.bar(x, acc_uni_042,color='#E9967A',width=w,label='ldp 0.42')
x2=x
plt.plot(x1, acc_perS, color='r', marker='s')
# plt.plot(x1, acc_perS, color='chocolate', marker='s')
# plt.plot(x2, acc_uniS, color='orange', marker='s', linestyle='--')
# plt.plot(x1, acc_perS, color='darkgreen', marker='s')
plt.plot(x2, acc_uniS, color='g', marker='s', linestyle='--')

plt.xlabel('$C$')
plt.ylabel('Test accuracy')
plt.title('Local privacy: Uniform2, $\delta_s=10^{-8}, n=10000, d=7850$', fontsize=9) # \\varepsilon^l_i \\in [0.1, 1]$
# plt.yscale('log')
plt.legend()
plt.show()
plt.savefig('/home/liuyixuan/workspace/personalShuffle/perS/utils/acc_with_C.png', dpi=600)# 设置dpi并保存图像到本地
plt.close()