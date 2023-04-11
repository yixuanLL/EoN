import pandas as pd
import matplotlib.pyplot as plt 
plt.switch_backend('agg')

acc_LDP_min = [0.56119631,0.56119631,0.56119631,0.56119631,0.56119631,0.56119631]
acc_PLDP = [0.756492843,0.637781186,0.677556237,0.775408998,0.64805726,0.738445808]
acc_uniS = [0.756492843,0.637781186,0.677556237,0.775408998,0.64805726,0.738445808] #,0.788445808,0.638957055,0.801738241]
acc_perS = [0.787730061,0.783384458,0.790644172,0.796728016,0.792944785,0.799744376] #,0.800613497,0.794325153,0.800664622]
acc_perSS = [0.757566462,0.566411043,0.779192229,0.781390593,0.734253579,0.761196319018404] #, 0.799539877, 0.779192229,0.79422290388548]
epsdist_list = ['Uni-\nform1','Gau-\nss1','Mix-\nGauss1','Uni-\nform2','Gau-\nss2','Mix-\nGauss2'] #,'Uni-\nform3','Gau-\nss3','Mix-\nGauss3']
x = list(range(len(epsdist_list)))
w = 0.15


plt.bar(x, acc_LDP_min,color='c',width=w,label='LDP-min') #chocolate blueviolet lightskyblue mediumslateblue
for i in range(len(x)):
    x[i] += w
plt.bar(x, acc_PLDP,color='mediumseagreen',width=w,label='PLDP') #chocolate dodgerblue cornflowerblue lightseagreen deepskyblue
for i in range(len(x)):
    x[i] += w
plt.bar(x, acc_uniS,color='yellowgreen',width=w,label='UniS') #moccasin yellowgreen
for i in range(len(x)):
    x[i] += w
plt.bar(x, acc_perS,color='orange',width=w,label='APES',tick_label = epsdist_list) #orange
for i in range(len(x)):
    x[i] += w
plt.bar(x, acc_perSS,color='orangered',width=w,label='S-APES') #chocolate


# plt.xlabel('$$')
plt.ylabel('Test accuracy')
plt.ylim(0,0.85)
plt.title('$\delta_s=10^{-8}, n=10000, d=7850, C=0.1$', fontsize=9) # \\varepsilon^l_i \\in [0.1, 1]$
# plt.yscale('log')
plt.legend(loc='lower right', fontsize=9)
plt.show()
plt.savefig('/home/liuyixuan/workspace/personalShuffle/perS/utils/acc_diffeps.png', dpi=600)# 设置dpi并保存图像到本地
plt.close()