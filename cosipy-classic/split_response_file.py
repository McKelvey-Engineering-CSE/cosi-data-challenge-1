# Split a response .npz file into two files: one containing the
# grid (the biggest item by far), and one containing the rest of
# the data.  The grid is stored as a sparse array to improve loading
# time and reduce space usage.
#
# For a file 'foo.npz', we create two new files
# 'foo_grid.npz' and 'foo_rest.npz'.

import sys
import os.path
import sparse
import numpy as np

# input argument: name of file to be split
fname = sys.argv[1]

basename, ext = os.path.splitext(fname)

with np.load(fname) as content:
    response_grid_normed = content['ResponseGrid']
    e_cen = content['e_cen']
    e_wid = content['e_wid']
    e_edges = content['e_edges']
    e_max = content['e_max']
    e_min = content['e_min']
    n_e = content['n_e']
    l_cen = content['l_cen']
    l_wid = content['l_wid']
    l_edges = content['l_edges']
    l_max = content['l_max']
    l_min = content['l_min']
    n_l = content['n_l']
    b_cen = content['b_cen']
    b_wid = content['b_wid']
    b_edges = content['b_edges']
    b_max = content['b_max']
    b_min = content['b_min']
    n_b = content['n_b']
    L_ARR = content['L_ARR']
    B_ARR = content['B_ARR']
    L_ARR_edges = content['L_ARR_edges']
    B_ARR_edges = content['B_ARR_edges']
    dL_ARR = content['dL_ARR']
    dB_ARR = content['dB_ARR']
    dL_ARR_edges = content['dL_ARR_edges']
    dB_ARR_edges = content['dB_ARR_edges']
    dOmega = content['dOmega']

grid = sparse.COO(response_grid_normed)

sparse.save_npz(basename + '_grid', grid)

np.savez_compressed(basename + '_rest', 
                    e_cen=e_cen, e_wid=e_wid, e_edges=e_edges, e_max=e_max, e_min=e_min, n_e=n_e, 
                    l_cen=l_cen, l_wid=l_wid, l_edges=l_edges, l_max=l_max, l_min=l_min, n_l=n_l, 
                    b_cen=b_cen, b_wid=b_wid, b_edges=b_edges, b_max=b_max, b_min=b_min, n_b=n_b,
                    L_ARR=L_ARR, B_ARR=B_ARR, L_ARR_edges=L_ARR_edges, B_ARR_edges=B_ARR_edges, 
                    dL_ARR=dL_ARR, dB_ARR=dB_ARR, dL_ARR_edges=dL_ARR_edges, dB_ARR_edges=dB_ARR_edges,
                    dOmega=dOmega)
