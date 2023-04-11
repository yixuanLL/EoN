import pickle
import numpy as np
import math


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def iid_divide(l, g):
    '''
    divide list l among g groups
    each group has either int(len(l)/g) or int(len(l)/g)+1 elements
    returns a list of groups
    '''
    num_elems = len(l)
    group_size = int(len(l)/g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l[group_size*i:group_size*(i+1)])
    bi = group_size*num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi+group_size*i:bi+group_size*(i+1)])
    return glist

def sparsify(updates, topk):
    '''
    return sparsified updates, with non-top-k as zeros
    '''
    d = updates.size
    non_top_idx = np.argsort(np.abs(updates))[:d-topk]
    updates[non_top_idx] = 0
    return updates

def topindex(updates, topk):
    '''
    return top=k indexes
    '''
    d = updates.size
    return np.argsort(np.abs(updates))[d-topk:]

def clip(updates, threshold):
    '''
    clip updates vector with L2 norm threshold
    input
        updates: 1-D vector
        threshold: L2 norm
    
    return:
        clipped 1-D vector
    '''

    # L2 norm
    L2_norm = np.linalg.norm(updates, 2)
    if L2_norm > threshold:
        updates = updates * (threshold * 1.0 / L2_norm)

    # # threshold for each dimension
    # updates = np.clip(updates, -threshold, threshold)
    return updates


def quantize(updates, k, x_max):
    B = np.zeros(k)
    for r in range(k):
        B[r] = -x_max + (2 * r * x_max) / (k - 1)

    B[0] = B[0]-1e-6
    B[k-1] = B[k-1]+1e-6

    for i in range(len(updates)):
        for j in range(k-1):
            if B[j] <= updates[i] and updates[i] <= B[j+1]:
                break
        # unbiased stochastic quantization
        try:
            updates[i] = np.random.choice(np.array(
                [B[j], B[j+1]]), p=[(B[j+1]-updates[i])/(B[j+1]-B[j]), (updates[i]-B[j])/(B[j+1]-B[j])])
        except RuntimeWarning:
            print("b_")  # TODO(tbd)

    return updates

def discrete(x, b):
    '''
    discrete value x \in [0, 1] to the discrete domain [b]
    x_bar = floor(x*k) + Ber(x*k - floor(x*k))
    '''
    xk = np.floor(v*b)
    r = np.random.rand()
    if r < (x*k - xk):
        return xk + 1
    else:
        return xk

# def shape_back(flattened_queried):
#     queried_weights = []
#     queried_weights.append(np.reshape(flattened_queried[:7840], (784, 10)))
#     queried_weights.append(flattened_queried[7840:])
#     return queried_weights


def transform(v, left, right, new_left, new_right):
    '''
    transform a vector/value from [left, right] to [new_left, new_right]
    '''
    # assert min(v) == left and max(v) == right, "ERROR: Please check the value range for the transforming input"  #TODO(0821): v is value or vector, min and v<left
    return new_left + (new_right - new_left)*(v - left)/(right - left)


def binarySearch (arr, l, r, x): 
    if r >= l: 
        mid = int(l + (r - l)/2)

        if arr[mid] == x: 
            return mid 
        elif arr[mid] > x: 
            return binarySearch(arr, l, mid-1, x) 
        else: 
            return binarySearch(arr, mid+1, r, x) 
  
    else: 
        return l