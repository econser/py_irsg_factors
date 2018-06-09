import os
import numpy as np
import sklearn.linear_model as lm
import numpy_to_latex as ltx



def to_csv(col_names, row_names, row_data):
    str = ''
    
    n_cols = len(col_names)
    for c_ix, c in enumerate(col_names):
        str += c
        if c_ix < n_cols-1:
            str += ', '
    str += '\n'
    
    for r_ix, r in enumerate(row_names):
        str += r + ', '
        for c_ix in range(0, n_cols-1):
            str += '{:0.3f}'.format(row_data[r_ix][c_ix])
            if c_ix < n_cols-2:
                str += ', '
        str += '\n'
    return str



def read_factor_file(fname):
    with open(fname, 'rb') as f:
        header = f.readline()
        header = header.split(', ')
        header = np.array(header, dtype=np.str)
    factors = np.genfromtxt(fname, delimiter=',', skip_header=1)
    return header, factors



#===============================================================================
# MAIN
#
if __name__ == '__main__':
    csv_base_path = '/home/econser/research/py_irsg_factors/output/' # TODO: parameterize
    
    query_ixs = np.arange(150)
    image_ixs = np.arange(1000)
    results = []
    
    for query_ix in query_ixs:
        fname = os.path.join(csv_base_path, 'q{:03}_factors_simple.csv'.format(query_ix))
        header, factors = read_factor_file(fname)
        
        obj_ixs = [i for i, hdr in enumerate(header) if 'OBJ|' in hdr]
        atr_ixs = [i for i, hdr in enumerate(header) if 'ATR|' in hdr]
        rel_ixs = [i for i, hdr in enumerate(header) if 'REL|' in hdr]
        
        o_factors = np.product(factors[:,obj_ixs], axis=1)
        a_factors = np.product(factors[:,atr_ixs], axis=1)
        r_factors = np.product(factors[:,rel_ixs], axis=1)
        
        or_factors = o_factors * r_factors
        oar_factors = np.product(factors[:,1::], axis=1)
        
        lr = lm.LinearRegression()
        
        o_oar_lr = lr.fit(o_factors.reshape(-1,1), oar_factors.reshape(-1,1))
        o_oar_r2 = o_oar_lr.score(o_factors.reshape(-1,1), oar_factors.reshape(-1,1))
        
        o_or_lr = lr.fit(o_factors.reshape(-1,1), or_factors.reshape(-1,1))
        o_or_r2 = o_or_lr.score(o_factors.reshape(-1,1), or_factors.reshape(-1,1))
        
        results.append((query_ix, o_or_r2, o_oar_r2))
    
    results = np.array(results)
    col_names = ['metrics', 'obj to obj-rel', 'obj to obj-atr-rel']
    row_names = ['min', 'max', 'median', 'average']
    
    average_r2 = np.average(results, axis=0)[1:]
    median_r2 = np.median(results, axis=0)[1:]
    min_r2 = np.min(results, axis=0)[1:]
    max_r2 = np.max(results, axis=0)[1:]
    row_data = np.vstack((min_r2, max_r2, median_r2, average_r2))
    
    tbl = ltx.to_latex(row_data)
    print tbl
    
    csv = to_csv(col_names, row_names, row_data)
    print csv
