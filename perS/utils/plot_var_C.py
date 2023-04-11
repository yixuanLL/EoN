from this import d
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
 
def clip_laplace(grad, epsilon, Clip_bound):
    C = Clip_bound
    b = 2*C/epsilon
    u = grad
    exp_part = np.exp((-C-u)/b)
    S = 1 - 0.5 * np.exp((-C+grad)/b) - 0.5 * exp_part
    p = np.random.uniform(0, 1, grad.shape[0])
    sp = S*p
    
    step_point = np.sign(sp - (0.5 - 0.5*exp_part))
    X = - step_point * b *np.log(1- 2*np.abs(sp - 0.5 + 0.5*exp_part))

    return X
 
def lap_var(vec, epsilon, clip_C):
    lambda_ = 2.0 * clip_C / epsilon
    noise_vec = np.random.laplace(loc=0, scale=lambda_, size=vec.shape)
    # return np.var(noise_vec)
    return np.linalg.norm(noise_vec)
 
def clipLap_var(vec, epsilon, clip_C):
    noise_vec = clip_laplace(vec, epsilon, clip_C)
    return np.linalg.norm(noise_vec)

epsilon = 0.3
clip_C = 0.01
vec = np.random.uniform(-clip_C, clip_C, 7850)
# epsilon = np.random.uniform(0, 1.0, 7850)

# print(clipLap_var(vec, epsilon, clip_C))
for epsilon in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]:
    for clip_C in [0.01, 0.05, 0.1, 0.25, 0.5, 0.7, 1.0]:
    # for clip_C in [0.01]:
        lap_var_list = []
        clipLap_var_list = []
        for i in range(10):
            lap_var_list.append(lap_var(vec, epsilon, clip_C))
            clipLap_var_list.append(clipLap_var(vec, epsilon, clip_C))
        lapVar = np.mean(lap_var_list)
        clipLapVar = np.mean(clipLap_var_list)
        # print('epsilon:{}, clip_C:{}, dist:uniform, lap var:{}, clipLap var:{}'.format(epsilon, clip_C, lapVar, clipLapVar))
        # print(epsilon, clip_C, lapVar, clipLapVar)
 
 
 
#读取csv文件
df = pd.read_csv('/home/liuyixuan/workspace/personalShuffle/RucPriv/utils/data/l2norm_lap.csv')
# rounds = df.loc[:, 'N']
color = ['r-.','r', 'b-.', 'b', 'g-.', 'g', 'c-.', 'c','k-.', 'k','m-.', 'm','y-.', 'y']
plt.switch_backend('agg')

p=[]
label_list = ['Lap e=.01', 'clipLap e=.01', 'Lap e=.05', 'clipLap e=.05','Lap e=.1', 'clipLap e=.1','Lap e=.3', 'clipLap e=.3','Lap e=.5', 'clipLap e=.5','Lap e=1', 'clipLap e=1',]
eps = ['0.01','0.05','0.1','0.3','0.5','1.0']
for i in range(0,len(df.columns),4):
    if i>=16:
        continue
    e = df.columns[i]
    eps = df.loc[:,e]
    n = df.columns[i+1]
    norm = df.loc[:,n]
    l_v = df.columns[i+2]
    lap_v = np.log(df.loc[:, l_v])#/np.log(100)
    c_v = df.columns[i+3]
    cliplap_v = np.log(df.loc[:, c_v])#/np.log(100)

    p1, = plt.plot(norm, lap_v, color[int(i/2)])
    p2, = plt.plot(norm, cliplap_v, color[int(i/2)+1])
    p.append(p1)
    p.append(p2)

    # plt.ylabel('Varience')
    # plt.xlabel('Clip Bound')
    # plt.title('Varience of Clip-Lap & Laplace, eps={}'.format(eps[int(i/4)]), fontsize='large')    
    # plt.show()
    # plt.savefig('/home/liuyixuan/workspace/personalShuffle/RucPriv/utils/var_C_{}.png'.format(eps[int(i/4)]), dpi=600)# 设置dpi并保存图像到本地
    # plt.close()
    i += 4

# 设置xy坐标范围
# plt.xlim((0,1.))
# plt.ylim((0.,100.))
 
#xy描述
plt.ylabel('Log(Varience)')
plt.xlabel('Clip Bound')

plt.legend(handles=[pi for pi in p], labels=[l for l in label_list], loc='best', fontsize=8)
plt.title('L2 Norm of Clip-Lap & Laplace noises', fontsize='large', fontweight='bold')
plt.show()
plt.savefig('/home/liuyixuan/workspace/personalShuffle/RucPriv/utils/l2norm_C.png', dpi=600)# 设置dpi并保存图像到本地
plt.close()