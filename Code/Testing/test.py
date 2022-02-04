import theano
import numpy as np
import pydot
from numba import jit, prange
# x = theano.tensor.fvector('x')
# target = theano.tensor.fscalar('target')
#
# W = theano.shared(numpy.asarray([0.2, 0.7]), 'W')
# y = (x * W).sum()
#
# cost = theano.tensor.sqr(target - y)
# gradients = theano.tensor.grad(cost, [W])
# W_updated = W - (0.1 * gradients[0])
# updates = [(W, W_updated)]
#
# f = theano.function([x, target], y, updates=updates)
#
# for i in range(10):
#     output = f([1.0, 1.0], 20.0)
#     print(output)

#
# x = theano.tensor.fvector('x')
# W = theano.shared(numpy.asarray([0.2, 0.7]), 'W')
# y = (x * W).sum()
#
# f = theano.function([x], y)
# theano.printing.pydotprint(f, outfile="f.png", var_with_name_simple=True)

from numba import jit
# import math
#
# @jit
# def hypot(x, y):
#     # Implementation from https://en.wikipedia.org/wiki/Hypot
#     x = abs(x);
#     y = abs(y);
#     t = min(x, y);
#     x = max(x, y);
#     t = t / x;
#     return x * math.sqrt(1+t*t)
#
# print(hypot(3,4))

# @jit(parallel=True)
# def test(x):
#     n = x.shape[0]
#     a = np.sin(x)
#     b = np.cos(a * a)
#     acc = 0
#     for i in prange(n - 2):
#         for j in prange(n - 1):
#             acc += b[i] + b[j + 1]
#     return acc
#
# test(np.arange(10))
#
# test.parallel_diagnostics(level=4)
#
#

import math
import numpy as np

from numba import cuda, jit, prange, vectorize, guvectorize
from sys import getsizeof
from multiprocessing import cpu_count, Pool



#
# n = 1_000
# k = 1_000
#
# coord1 = np.zeros((n, 2), dtype=np.float32)
# coord2 = np.zeros((k, 2), dtype=np.float32)
#
# coord1[:,0] = np.random.uniform(-90, 90, n).astype(np.float32)
# coord1[:,1] = np.random.uniform(-180, 180, n).astype(np.float32)
# coord2[:,0] = np.random.uniform(-90, 90, k).astype(np.float32)
# coord2[:,1] = np.random.uniform(-180, 180, k).astype(np.float32)
#
# coord1 = np.sort(coord1,axis=0)
# coord2 = np.sort(coord2,axis=0)
#
#
# def get_nearby_mp(coord1, coord2, max_dist):
#     cores = cpu_count()
#     coord1_split = np.array_split(coord1, cores)
#     starmap_args = [(split, coord2, 1.0) for split in coord1_split]
#     with Pool(cores) as p:
#         data = np.concatenate(p.starmap(get_nearby_py, starmap_args))

import pandas as pd


# =============================================================================
# Dummy transactional database generation
# =============================================================================
def generate_db(sku_num: int,
                day_num: int,
                row_num: int,
                sale_max: int) -> tuple:
    """
    Generate dummy transactional database
    Every row records number of pieces sold per each SKU in a specific day
    Parameters
    ----------
    sku_num : pd.core.frame.DataFrame
        Number of SKUs
    day_num : np.ndarray
        Number of days
    row_num : np.ndarray
        Number of rows contained in the database
    sale_max : np.ndarray
        Maximum number of pieces sold per transaction
    Returns
    -------
    tx_db, day, sku_id, sales : tuple
        tx_db: pd.core.frame.DataFrame
            Transactional database
        day: list
            day in which sale occurred
        sku_id: list
            SKU unique identifier
        sales: list
            Number of pieces sold
    """
    # Reset the random seed
    np.random.seed(seed=22)
    # Random generation of day vector
    day = np.random.randint(low=1,
                            high=day_num + 1,
                            size=row_num,
                            dtype='int64')
    # Random generation of day vector
    sku_id = np.random.randint(low=1,
                               high=sku_num + 1,
                               size=row_num,
                               dtype='int64')
    # Random generation of day vector
    sales = np.random.randint(low=0,
                              high=sale_max + 1,
                              size=row_num,
                              dtype='int64')
    # Pack vectors in a single DataFrame
    tx_db = pd.DataFrame(data={'day': day, 'sku_id': sku_id, 'sales': sales},
                         columns=['day', 'sku_id', 'sales'])
    # Consolidate database's sales and sort by product and days
    tx_db = tx_db.groupby(by=['day', 'sku_id'], axis=0).sum().reset_index()

    # Return the DBs
    return tx_db, day, sku_id, sales

# =============================================================================
# Generate dummy transactional db
# =============================================================================
tx_db, day, sku_id, sales = generate_db(sku_num=1000,
                                        day_num=30,
                                        row_num=10000,
                                        sale_max=10)

print(generate_db.parallel_diagnostics(level=4))
print(tx_db, day, sku_id, sales)










