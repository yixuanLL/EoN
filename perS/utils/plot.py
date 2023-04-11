import pandas as pd
import matplotlib.pyplot as plt
 
#读取csv文件
# df = pd.read_csv('/home/liuyixuan/workspace/personalShuffle/RucPriv/utils/data/n50_005~015_C.csv')
# df = pd.read_csv('/home/liuyixuan/workspace/personalShuffle/RucPriv/utils/data/n50_norm0.1_epsn_thre.csv')
df = pd.read_csv('/home/liuyixuan/workspace/personalShuffle/perS/utils/data/perLeft_n1w_eps.csv') # cross_silo & cross_device
rounds = df.loc[:, 'perLeft']
color = ['r', 'b-.', 'g-.', 'm', 'r', 'c', 'k-.', 'c-.']
plt.switch_backend('agg')

for i in range(len(df.columns)):
    if i==0:
        continue
    exp = df.columns[i]
    y = df.loc[:,exp]
    plt.plot(rounds, y, color[i])

# 设置xy坐标范围
# plt.xlim((1000,100000))
# plt.ylim((0.,0.85))
 
#xy描述
# plt.ylabel('acc')
# plt.xlabel('round')
plt.ylabel('Central epsilon')
plt.xlabel('max Local epsilon')

plt.legend(loc='best', fontsize=8)
# plt.title('MNIST with same DP 2.5', fontsize='large', fontweight='bold')
# plt.title('MNIST with personal LDP(0.05~0.15)', fontsize='large', fontweight='bold')
# plt.title('Central DP with n (LDP=[0.1,1])', fontsize='large', fontweight='bold')
plt.title('Central DP with min LDP (uniform ~1)', fontsize='large', fontweight='bold')
plt.show()
plt.savefig('/home/liuyixuan/workspace/personalShuffle/perS/utils/perLeft_n1w_eps.png', dpi=600)# 设置dpi并保存图像到本地
plt.close()