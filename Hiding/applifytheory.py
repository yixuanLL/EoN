# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.

import computeamplification_perS as CA_perS
import computeamplification as CA_uniS
import numpy as np
from scipy.special import comb, gamma

############### for perS
#number of iterations of binary search. The higher T is, the more accurate the result
num_iterations = 10
# #This is a parameter of the empirical analysis computation that can be tuned for efficiency. The larger step is, the less accurate the result, but more efficient the algorithm.
step = 100

n = 1 * 10**4
# epsorig = np.random.uniform(1,1, n)
epsorig = np.array([1]*n)
delta = 10**(-10)

eps_max = np.max(epsorig)

class Clones:
    """Base class for "privacy amplification by shuffling" bounds."""

    def __init__(self, name='BoundBase', num_interations=10, step=100):
        self.name = name
        self.num_interations = num_interations
        self.step = step

    def get_name(self, with_mech=False):
        return self.name

class UniS(Clones):
    """Implement the bound from Clones et al. [FMT'21]"""

    def __init__(self, name='FMT\'21'):
        super(UniS, self).__init__(name=name)
        # The constants in the bound are only valid for a certain parameter regime
        
    def get_eps(self, eps, n, delta):
        eps_max = np.max(eps)
        try:
            numerical_upperbound = CA_uniS.numericalanalysis(n, eps_max, delta, self.num_interations, self.step, True)
        except AssertionError:
            return eps_max #np.nan
        return numerical_upperbound

    
class PerS(Clones):
    """Implement the bound from Erlignsson et al. [SODA'19]"""

    def __init__(self, name='EoN'):
        super(PerS, self).__init__(name=name)
        # The constants in the bound are only valid for a certain parameter regime
        
    def get_eps(self, eps, n, delta):
        try:
            numerical_upperbound = CA_perS.numericalanalysis(n, eps, delta, self.num_interations, self.step, True)
        except AssertionError:
            return np.max(eps) #np.nan
        return numerical_upperbound
    
    
class RDP(Clones):
    """Implement the bound from Erlignsson et al. [SODA'19]"""

    def __init__(self, name='GDDTK\'21'):
        super(RDP, self).__init__(name=name)
        # The constants in the bound are only valid for a certain parameter regime
    def get_eps(self, eps, n, delta):
        eps = np.max(eps)
        dp_upperbound_min = eps
        try:
            n_bar = int((n-1)/(2*np.exp(eps)) + 1)
            for lambd in range(2,5000,50):
                sum = 0
                for i in range(2, lambd+1, 1):   
                    sum += comb(lambd, i) * i * gamma(i/2.0) * ((np.exp(2*eps)-1)**2/(2*np.exp(2*eps)*n_bar))**(i/2.0)
                rdp_upperbound = 1 / (lambd-1) * np.log(1+ comb(lambd, 2)* ((np.exp(eps)-1)**2)/(n_bar*np.exp(eps)) + sum \
                    + np.exp(eps*lambd-(n-1)/(8*np.exp(eps)))) # upperbound1
                dp_upperbound = self.rdp2dp(rdp_upperbound, lambd, delta)
                if dp_upperbound_min > dp_upperbound:
                    dp_upperbound_min = dp_upperbound
        except AssertionError:
            return eps
       
        return dp_upperbound_min
    
    def rdp2dp(self, rdp_e, lambd, delta):
        return rdp_e + (np.log(1/delta)+(lambd-1)*np.log(1-1/lambd)-np.log(lambd))/(lambd-1)

