#Functions for analyzing results.

import numpy as np

def make_frequency_table(datalist, n=None, start=None, end=None, norm=True):
    """function for construct frequency table of data"""
    if start == None :
        start = min(datalist)//1
    if end == None :
        end = max(datalist)//1+1
    if n == None :
        width = 1
    else :
        width = (end-start)/n

    x = [start + width*(i+0.5) for i in range(-1, n+1)]

    y = [0 for i in range(n+2)] # y[0] and y[-1] are padding. (The value of them should be 0.)
    
    for data in datalist :
        if start <= data < end :
            y[int((data-start)/width) + 1] += 1
    
    if norm :
        #To represent frequency table to distribution graph, we should normalize the table.
        norm_constant = width*sum(y)
        return np.array(x), np.array(y)/norm_constant
    
    return np.array(x), np.array(y)
