from blanket.shuffleddp.mechanisms import *
from blanket.shuffleddp.amplification_bounds import *
import matplotlib.pyplot as plt
from applifytheory import *
import numpy as np


delta = 10**(-8)

ns = np.geomspace(1000, 1000000, num=50, dtype=int)
# ns=[1000,10000]
def plot_panel(xs, bounds):
    fig = plt.figure()
    ls = ['--', ':', '-.', '--','--','--','--', '-', '-', '-']
    m = ['', '', '', '','', '', '', 'o', 'p', '*']
    c = ['slategrey', 'dodgerblue', 'blueviolet', 'darkcyan', 'yellowgreen', 'r', 'orange','yellowgreen', 'r', 'orange']
    i=-1
    for b in bounds:
        print('theory:', b.get_name())
        for dist in ['Uniform', 'Gauss', 'MixGauss']:
            print('dist:', dist)
            ys = list()
            for x in xs:
                eps0 = gen_eps(0.05, 1, x, dist)
                re = b.get_eps(eps0, x, delta)
                ys.append(re)
                print(x, '\t', re)     
            i+=1       
            if b.get_name() != 'EoN': 
                plt.plot(xs, ys, label=b.get_name(), linestyle=ls[i], marker=m[i], color=c[i], markevery=10)   
                break  
            else:
                me = 10
                if dist == 'MixGauss':
                    me = 8
                plt.plot(xs, ys, label=b.get_name()+' '+dist, linestyle=ls[i], marker=m[i], color=c[i], markevery=me) 
    plt.legend(loc='upper right')

def gen_eps(l, r, n, dist):
    if dist == 'Uniform':
        return np.random.uniform(l, r, n)
    elif dist == 'Gauss':
        eps0 = np.random.normal(0.1, 1, n)
        eps0 = np.maximum(eps0, l)
        eps0 = np.minimum(eps0, r)
        return eps0
    elif dist == 'MixGauss':
        step = int(n*0.9)
        eps_low = np.random.normal(0.1, 1, step)
        eps_high = np.random.normal(1, 1, n-step)
        eps0 = np.concatenate((eps_low, eps_high))
        eps0 = np.maximum(eps0, l)
        eps0 = np.minimum(eps0, r)
        return eps0
    elif dist == 'Single':
        eps0 = np.array([l]*n)
        return eps0
    else:
        return 0


bounds = [Erlingsson(),
          Hoeffding(LDPMechanism()),
          RDP(),
          UniS(),
          PerS()]


plt.switch_backend('agg')

## calculate bounds
plot_panel(ns, bounds)

delta = 10**(-10)
plt.xlabel('$n$')
plt.ylabel('$\\varepsilon^c$')
plt.title('$\\varepsilon^l_i \\in [0.05, 1], \\delta_s = 10^{}$'.format('{-%d}' % np.log10(1/delta)))
plt.xscale('log')
plt.yscale('log')
plt.show()
plt.savefig('./epsilon.png', dpi=600)
plt.close()



     
