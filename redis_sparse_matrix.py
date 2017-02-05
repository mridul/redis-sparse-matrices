from itertools import izip

import numpy as np
from scipy.sparse.base import isspmatrix
from scipy.sparse.sputils import isintlike, IndexMixin


class redis_sparse_matrix(IndexMixin):
    def __init__(self, redis, key, shape=None, init_arr=None):
        self.redis = redis
        self.key = 'rsm:{}'.format(key)
        if init_arr is not None:
            if not isspmatrix(init_arr):
                raise TypeError('init_arr may only be a scipy sparse matrix')

            self.shape = init_arr.shape
            self._init_using_arr(shape, init_arr)
        else:
            self.shape = shape


    def __getitem__(self, index):
        """
        if key=(i,j), return the corresponding element.
        if either i or j is absent, return sparse matrix with just these
        """

        # TODO slicing if needed

        i, j = self._unpack_index(index)
        i_intlike, j_intlike = isintlike(i), isintlike(j)

        if i_intlike and j_intlike:
            if i < 0 or j < 0:
                raise IndexError('index out of range')

            return self._get_element(i, j)


    def __setitem__(self, index, x):
        i, j = self._unpack_index(index)
        return self._set_element(i, j, x)


    def _get_element(self, i, j):
        index_accessor = self._index_accessor(i, j)
        return self.redis.hget(self.key, index_accessor)


    def _index_accessor(self, i, j):
        return '({},{})'.format(i, j)


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
