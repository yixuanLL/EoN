# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.

from os import stat
from joblib import parallel_backend
import scipy.stats as stats
import math
import numpy as np
# from poibin import PoiBin
import time
# from numba import vectorize, njit, guvectorize,jit, cuda

# This document contains 4 computations: 2 empirical and 2 theoretical.
# 1. Empirical analysis
# 2. Theoretical analysis


# ========= SUPPORT FUNCTIONS ==========
# Algo4
# This function uses binary search to approximate the smallest eps such that deltacomp will output something smaller than delta (i.e. an algorithm is (eps, delta)-DP)
def binarysearch(deltacomp, delta, num_iterations, epsupper, eps1):
    '''
    binary search to find min epsilon such that deltacomp(epsilon)<delta
    deltacomp = function that takes epsilon as input and outputs delta
    num_iterations = number of iterations, accuracy is 2^(-num_iterations)*epsupper
    epsupper = upper bound for epsilon. You should be sure that deltacomp(epsupper)<delta.
    '''
    llim = 0
    rlim = epsupper
    for t in range(num_iterations):
        mideps = (rlim + llim) / 2
        delta_for_mideps = deltacomp(mideps, delta, eps1)
        if delta_for_mideps < delta:
            llim = llim
            rlim = mideps
        else:
            llim = mideps
            rlim = rlim
    return rlim

# ================/EXACT EMPIRICAL ANALYSIS WITH STEPS - SAMPLING EMPIRICAL/==============

#This a subroutine in the main algorithm5.
def onestep(c, eps, eps1, pminusq):
    '''
    onestep computes the e^(eps)-divergence between p=alpha*Bin(c,0.5)+(1-alpha)*(Bin(c,1/2)+1) and q=alpha*(Bin(c,0.5)+1)+(1-alpha)*Bin(c,1/2), where alpha=e^(eps)/(1+e^(eps))
    if pminusq=True then computes D_(e^eps)(p|q), else computes D_(e^eps)(q|p)
    '''
    # alpha = math.exp(eps0) / (math.exp(eps0) + 1) #q，depends on R(x_1)
    alpha = math.exp(eps1) / (math.exp(eps1) + 1)
    effeps = math.log(((math.exp(eps) + 1) * alpha - 1) / ((1 + math.exp(eps)) * alpha - math.exp(eps))) #eps_q,eps
    if pminusq == True:
        beta = 1 / (math.exp(effeps) + 1)
    else:
        beta = 1 / (math.exp(-effeps) + 1)
    cutoff = beta * (c + 1)
    pconditionedonc = (alpha * stats.binom.cdf(cutoff, c, 0.5) + (1 - alpha) * stats.binom.cdf(cutoff - 1, c, 0.5))
    qconditionedonc = ((1 - alpha) * stats.binom.cdf(cutoff, c, 0.5) + alpha * stats.binom.cdf(cutoff - 1, c, 0.5))
    if pminusq == True:
        return (pconditionedonc - math.exp(eps) * qconditionedonc)
    else:
        return ((1 - qconditionedonc) - math.exp(eps) * (1 - pconditionedonc))


def deltacomp(n, expectation, sigma, gamma, eps1, eps, deltaupper, step, upperbound = True):
    '''
    Let C=Bin(n-1, e^(-eps0)) and A=Bin(c,1/2) and B=Bin(c,1/2)+1 and alpha=e^(eps0)/(e^(eps0)+1)
    p samples from A w.p. alpha and B otherwise
    q samples from B w.p. alpha and A otherwise
    deltacomp attempts to find the smallest delta such P and Q are (eps,delta)-indistinguishable, or outputs deltaupper if P and Q are not (eps, deltaupper)-indistinguishable.
    If upperbound=True then this produces an upper bound on the true delta (except if it exceeds deltaupper), and if upperbound=False then it produces a lower bound.
    '''
    deltap = 0  # this keeps track of int max{0, p(x)-q(x)} dx
    deltaq = 0  # this keeps track of int max{0, q(x)-p(x)} dx
    probused = 0  # To increase efficiency, we're only to search over a subset of the c values.
    # This will keep track of what probability mass we have covered so far.

    # Now, we are going to iterate over the n/2, n/2-step, n/2+step, n/2-2*steps, ...
    for B in range(1, int(np.ceil(n/step)), 1):
        for s in range(2):
            if s == 0:
                if B==1:
                    upperc = int(np.ceil(expectation+B*step))  # This is stepping up by "step". 从c的期望附近开始查找
                    lowerc = upperc - step
                else:
                    upperc = int(np.ceil(expectation + B * step))  # This is stepping up by "step".
                    lowerc = upperc - step + 1
                if lowerc>n-1: #判断lowerc是否超过最大实验成功次数n-1，没有超过就可以计算
                    inscope = False
                else:
                    inscope = True
                    upperc = min(upperc, n-1)
            if s == 1:
                lowerc = int(np.ceil(expectation-B*step))
                upperc = lowerc + step - 1
                if upperc<0:
                    inscope = False
                else:
                    inscope = True
                    lowerc = max(0, lowerc)

            if inscope == True: 
                p = expectation / n
                cdfinterval = stats.norm.cdf(upperc, expectation-0.5, sigma) -  stats.norm.cdf(lowerc, expectation-0.5, sigma) #+ stats.norm.pmf(lowerc, expectation-0.5, sigma) # approximate to normal distribution
                # approx to refined normal (RNA)
                # cdfinterval = RNA((upperc+0.5-expectation)/sigma, gamma) - RNA((lowerc+0.5-expectation)/sigma, gamma)
            # This is the probability mass in the interval (in Bin(n-1, p))

                if max(deltap, deltaq) > deltaupper:
                    return deltaupper

                if 1 - probused < deltap and 1 - probused < deltaq:
                    if upperbound == True:
                        return max(deltap + 1 - probused, deltaq + 1 - probused)
                    else:
                        return max(deltap, deltaq)

                else:
                    deltap_upperc = onestep(upperc, eps, eps1, True) #calculate deltaP deltaQ
                    deltap_lowerc = onestep(lowerc, eps, eps1, True)
                    deltaq_upperc = onestep(upperc, eps, eps1, False)
                    deltaq_lowerc = onestep(lowerc, eps, eps1, False)

                    if upperbound == True:
                        # compute the maximum contribution to delta in the segment.
                        # The max occurs at the end points of the interval due to monotonicity
                        deltapadd = max(deltap_upperc, deltap_lowerc)
                        deltaqadd = max(deltaq_upperc, deltaq_upperc)
                    else:
                        deltapadd = min(deltap_upperc, deltap_lowerc)
                        deltaqadd = min(deltaq_upperc, deltaq_lowerc)

                    deltap = deltap + cdfinterval * deltapadd
                    deltaq = deltaq + cdfinterval * deltaqadd

                probused = probused + cdfinterval  # updates the mass of C covered so far

    return max(deltap, deltaq)




# #if UL=1 then produces upper bound, else produces lower bound.
def numericalanalysis(n, epsorig, delta, num_iterations, step, upperbound):
    '''
    Empirically computes the privacy guarantee of achieved by shuffling n eps0-DP local reports.
    num_iterations = number of steps of binary search, the larger this is, the more accurate the result
    If upperbound=True then this produces an upper bound on the true shuffled eps, and if upperbound=False then it produces a lower bound.
    '''
    # start = time.time()
    e1_idx = np.argmax(epsorig)
    expectation, sigma, gamma = probEoN(epsorig, e1_idx)
    # pb = PoiBin(pij)
    eps1 = np.max(epsorig)
    # in order to speed things up a bit, we start the search for epsilon off at the theoretical upper bound.
    if expectation >= 16*np.log(4/delta):
        # checks if this is a valid parameter regime for the theoretical analysis.
        # If yes, uses the theoretical upper bound as a starting point for binary search
        epsupper = closedformanalysis_perS(n, epsorig, expectation, delta)
        # print('============closedformanalysis_perS:', epsupper, '===========')
    else:
        epsupper = epsorig
        return eps1
        # print('============closedformanalysis_perS: WRONG ===========')

    def deltacompinst(eps, delta, eps1):
        return deltacomp(n, expectation, sigma, gamma, eps1, eps, delta, step, upperbound)

    return binarysearch(deltacompinst, delta, num_iterations, epsupper, eps1)


# ===========/THEORYL Clones/========
def closedformanalysis_uniS(n, epsorig, delta):
    '''
    Theoretical computation the privacy guarantee of achieved by shuffling n eps0-DP local reports.
    '''
    eps_max = np.max(epsorig)
    if eps_max > math.log(n / (16 * math.log(4 / delta))):
        print("This is not a valid parameter regime for this analysis")
        return epsorig
    else:
        a = 8 * (math.exp(eps_max) * math.log(4 / delta)) ** (1 / 2) / (n) ** (1 / 2)
        c = 8 * math.exp(eps_max) / n
        e = math.log(1 + a + c)
        b = 1 - math.exp(-eps_max)
        d = (1 + math.exp(-eps_max - e))
        return math.log(1 + (b / d) * (a + c))
    
    
# ===========/THEORYL EoN/========
def closedformanalysis_perS(n, epsorig, expectation, delta):
    '''
    Theoretical computation the privacy guarantee of achieved by shuffling n eps0-DP local reports.
    '''
    sum_p = expectation
    if sum_p < 16*np.log(4/delta):
        print("This is not a valid parameter regime for this analysis")
        return sum_p
    else:
        e1 = max(epsorig)
        sample_ratio = (np.exp(e1)-1)/(np.exp(e1)+1) # for x1
        exp_ei_prime = 1+sample_ratio * ( (8 * np.sqrt(np.log(4/delta)/sum_p)) + 8/sum_p)
        return np.log(exp_ei_prime)
    
    
# ========== pij EoN ===========

def probEoN(ei_arr, e1_idx):  
    n = len(ei_arr)
    ej = ei_arr
    mu = 0
    sigma = 0
    gamma = 0
    for i in range(len(ei_arr)):
        ei = ei_arr[i]
        if i == e1_idx:
            continue # x2~xn, ei!=e1
        pij = ei/ej * (1-np.exp(-ej))/(1-np.exp(-ei)) * np.exp(-np.maximum(ej, ei)) / n #aaai version
        mu += np.sum(pij) #aaai version
        sigma += np.sum(pij*(1-pij)) #aaai version
      
    sigma = np.sqrt(sigma)
    return mu, sigma, gamma




def RNA(x, gamma):
    G = stats.norm.cdf(x, 0, 1) + gamma * (1-x**2) * stats.norm.cdf(x, 0, 1) / 6
    return G

