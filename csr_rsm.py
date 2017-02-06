from itertools import izip

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.base import isspmatrix

from .rsm_base import redis_spmatrix


class redis_csr_matrix(redis_spmatrix):
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


    def _get_row(self, i):
        # column indices for row i are in indices[indptr[i]:indptr[i+1]]
        # data for row i is in data[indptr[i]:indptr[i+1]]
        # TODO indptr should be hash
        indptr = int(self.redis.lindex(self.indptr_key, i))
        indptr_1 = int(self.redis.lindex(self.indptr_key, i+1))
        data = self.redis.lrange(self.data_key, indptr, indptr_1-1)
        indices = self.redis.lrange(self.indices_key, indptr, indptr_1-1)

        data = np.array(data, dtype=self.dtype)
        indices = np.array(indices, dtype='int')
        indptr = [0, len(data)]

        m = csr_matrix((data, indices, indptr), shape=(1, self.shape[1]))

        return m

    def _get_element(self, i, j):
        index_accessor = self._index_accessor(i, j)
        return self.redis.hget(self.key, index_accessor)

    def _index_accessor(self, i, j):
        return '({},{})'.format(i, j)

    def _get_row_accessor(self, i):
        return '({}'.format(i)

    def _init_using_arr(self, shape, A):
        data = A.data.tolist()
        indptr = A.indptr.tolist()
        indices = A.indices.tolist()

        # overwrite existing keys
        self.redis.del(self.data_key, self.indptr_key, self.indices_key)

        self.redis.rpush(self.data_key, *data)
        self.redis.rpush(self.indptr_key, *indptr)
        self.redis.rpush(self.indices_key, *indices)

    def _set_element(self, i, j, x):
        index_accessor = self._index_accessor(i, j)
        return self.redis.hset(self.key, index_accessor, x)
