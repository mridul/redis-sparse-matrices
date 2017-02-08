from itertools import izip

import numpy as np
from scipy.sparse.base import isspmatrix
from scipy.sparse.sputils import isintlike, IndexMixin

from rsm_base import redis_spmatrix


class redis_sparse_matrix(redis_spmatrix):
    def __init__(
            self, redis, key, shape=None,
            init_arr=None, dtype='float64'
    ):
        self.redis = redis
        self.key = 'rsm:{}'.format(key)
        self.data_key = '{}:data'.format(self.key)
        self.indices_key = '{}:indices'.format(self.key)
        self.indptr_key = '{}:indptr'.format(self.key)
        if init_arr is not None:
            if not isspmatrix(init_arr):
                raise TypeError('init_arr may only be a scipy sparse matrix')

            self.shape = init_arr.shape
            self.dtype = init_arr.dtype
            self._init_using_arr(shape, init_arr)
        else:
            self.shape = shape
            self.dtype = dtype

    def _get_col(self, index):
        raise NotImplementedError

    def _get_element(self, i, j):
        index_accessor = self._index_accessor(i, j)
        return self.redis.hget(self.key, index_accessor)

    def _get_row(self, i):
        all_keys = self.redis.hkeys(self.key)
        rk = self._get_row_accessor(i)
        row_keys = [k for k in all_keys if rk in k]
        return self.redis.hmget(self.key, row_keys)


    def _index_accessor(self, i, j):
        return '({},{})'.format(i, j)


    def _get_row_accessor(self, i):
        return '({}'.format(i)


    def _init_using_arr(self, shape, A):
        # first build a dictionary representation of the array
        # then write it using hmset
        rows, cols = A.nonzero()
        dok = {}
        for i, j in izip(rows, cols):
            key = self._index_accessor(i, j)
            dok[key] = A[i, j]

        return self.redis.hmset(self.key, dok)

    def _set_element(self, i, j, x):
        index_accessor = self._index_accessor(i, j)
        return self.redis.hset(self.key, index_accessor, x)
