import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from calibration import Calibration


def function_12(e,C,u):
    u = np.clip(u,-C,C)

    b = 2*C/e
    S = 1-0.5*np.exp((-C+u)/b)-0.5*np.exp((-C-u)/b)
    E_n2 = 1/(2*S) * ((b**2 - (u+C+b)**2) * np.exp((-C-u)/b) + (b**2 - (-u+C+b)**2) * np.exp((-C+u)/b)) + 2*b*b
    E_n2_lap = 2*b*b
    E_n2 = np.mean(E_n2)
    E_n2_lap=np.mean(E_n2_lap)
    return E_n2, E_n2_lap

def lap_var():
    plt.switch_backend('agg')
    C_list =np.linspace(0.01,1,100)
    c = Calibration()
    n=1000

    eps=c.gen_eps(n, 'Uniform1')
    y1_list = []
    y2_list = []
    for C in C_list:   
        y1,y2 = function_12(eps, C, 0)
        y1_list.append(y1)
        y2_list.append(y2)
    plt.plot(C_list,y1_list,'r',label='CLap, Uniform1', markevery=15)
    plt.plot(C_list,y2_list,'r-.',label='Lap, Uniform1', markevery=15)
    plt.ylabel('$D[\~g]$')
    plt.xlabel('$C$')  

    eps=c.gen_eps(n, 'Uniform2')
    y1_list = []
    y2_list = []
    for C in C_list:   
        y1,y2 = function_12(eps, C, 0)
        y1_list.append(y1)
        y2_list.append(y2)
    plt.plot(C_list,y1_list,color='orange', marker='*',linestyle='-',label='CLap, Uniform2', markevery=15)
    plt.plot(C_list,y2_list,color='orange', marker='*',linestyle='-.',label='Lap, Uniform2', markevery=15)  

    eps=c.gen_eps(n, 'Gauss1')
    y1_list = []
    y2_list = []
    for C in C_list:   
        y1,y2 = function_12(eps, C, 0)
        y1_list.append(y1)
        y2_list.append(y2)
    plt.plot(C_list,y1_list,color='yellowgreen', marker='*',linestyle='-',label='CLap, Gauss1', markevery=15)
    plt.plot(C_list,y2_list,color='yellowgreen', marker='*',linestyle='-.',label='Lap, Gauss1', markevery=15)  

    eps=c.gen_eps(n, 'Gauss2')
    y1_list = []
    y2_list = []
    for C in C_list:   
        y1,y2 = function_12(eps, C, 0)
        y1_list.append(y1)
        y2_list.append(y2) 
    plt.plot(C_list,y1_list,color='darkcyan', marker='*',label='CLap, Gauss2', markevery=15)
    plt.plot(C_list,y2_list,color='darkcyan', marker='*',linestyle='-.',label='Lap, Gauss2', markevery=15)

    eps=c.gen_eps(n, 'MixGauss1')
    y1_list = []
    y2_list = []
    for C in C_list:   
        y1,y2 = function_12(eps, C, 0)
        y1_list.append(y1)
        y2_list.append(y2)
    plt.plot(C_list,y1_list,color='dodgerblue', marker='*',linestyle='-',label='CLap, MixGauss1', markevery=15)
    plt.plot(C_list,y2_list,color='dodgerblue', marker='*',linestyle='-.',label='Lap, MixGauss1', markevery=15) 

    eps=c.gen_eps(n, 'MixGauss2')
    y1_list = []
    y2_list = []
    for C in C_list:   
        y1,y2 = function_12(eps, C, 0)
        y1_list.append(y1)
        y2_list.append(y2)
    plt.plot(C_list,y1_list,color='mediumslateblue', marker='*',linestyle='-',label='CLap, MixGauss2', markevery=156)
    plt.plot(C_list,y2_list,color='mediumslateblue', marker='*',linestyle='-.',label='Lap, MixGauss2', markevery=15)  
    # plt.yscale("log") 
    # plt.xlim(0.01,1)

    plt.legend(loc='upper left', fontsize=8)
    plt.show()
    plt.savefig('/home/liuyixuan/workspace/personalShuffle/perS/lap_var.png', dpi=600)# 设置dpi并保存图像到本地
    plt.close()

def function_11(e,C,u): 
    u = np.clip(u,-C,C)
    uc = u
    b = 2*C/e
    S = 1-0.5*np.exp((-C+u)/b)-0.5*np.exp((-C-u)/b)
    # E_n = 1/(2*S) * ((b+C+u)*np.exp((-C-u)/b) - (b+C-u)*np.exp((-C+u)/b)) + u
    e1 = np.exp((-C-u)/b)
    e2 = np.exp((-C+u)/b)
    E_n = ((C+b)*(e1-e2)+2*u) / (2-e1-e2) - u

    return E_n #, E_n_cali

def lap_mean():
    C=0.1
    C_list=np.linspace(0.01,1,1000)

    cali = Calibration()
    # E_n = function_11(eps,C,-C)
    # print(E_n, E_n_cali)
    plt.switch_backend('agg')
    # plt.plot(C,E_n,'r', label="E_n")
    # plt.plot(C,E_n_cali,'g', label='E_n_cali')

    # eps_list = [0.1,0.5,1.0, 2.0]
    eps_name = ['CLap Uniform1', 'Lap Uniform1', 'CLap Uniform2', 'Lap Uniform2', 'CLap Gauss1', 'Lap Gauss1','CLap Gauss2', 'Lap Gauss2', 'CLap MixGauss1', 'Lap MixGauss1', 'CLap MixGauss2', 'Lap MixGauss2']
    eps_list = ['Uniform1', 'Uniform2', 'Gauss1', 'Gauss2', 'MixGauss1', 'MixGauss2']
    # x=np.random.normal(0,1,10000)
    x = np.linspace(-1,1,10000)

    ls = ['-', '--', '-', '--', '-', '--', '-', '--', '-', '--', '-', '--']
    m = ['*', '*', '', '', '+', '+', 'x', 'x', '.', '.', '.', 7]
    c = ['r', 'r', 'orange', 'orange','yellowgreen', 'yellowgreen','darkcyan', 'darkcyan',  'dodgerblue', 'dodgerblue', 'mediumslateblue', 'mediumslateblue']
    i=0


    for dist in eps_list:
        eps = cali.gen_eps(10000, dist)
        En_list=[]
        E_n_lap_list=[]
        for C in C_list:
            x = np.linspace(-C,C,10000)
            E_n= function_11(eps,C,x)
            # E_n = cali.E_noisy_mean_g(eps,x,C)
            En_list.append(np.mean(E_n))
            E_n_lap_list.append(0)
        print(dist, i)
        plt.plot(C_list,En_list,color=c[i],linestyle='-',label=eps_name[i], marker=m[i], markevery=100) #label="epsilon="+str(eps)
        plt.plot(C_list, E_n_lap_list, color=c[i+1],linestyle='-.',label=eps_name[i+1], marker=m[i+1], markevery=100)
        print(En_list[0], En_list[10], En_list[90])
        i+=2
        

    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.xlabel('$C$')
    plt.ylabel('$E[\~g - g]$')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()
    plt.savefig('/home/liuyixuan/workspace/personalShuffle/perS/lap_mean_6.png', dpi=600)# 设置dpi并保存图像到本地
    plt.close()

lap_mean()
# lap_var()