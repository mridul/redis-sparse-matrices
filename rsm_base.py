from abc import ABCMeta, abstractmethod

from scipy.sparse.sputils import IndexMixin, isintlike


class redis_spmatrix(IndexMixin):
    __metaclass__ = ABCMeta

    @abstractmethod
    def _get_col(self, index):
        raise NotImplementedError

    @abstractmethod
    def _get_element(self, i, j):
        raise NotImplementedError

    def _get_or_set_item(self, operation_type, index, item=None):
        operation_map = {
            'get': {
                'element': self._get_element,
                'row': self._get_row,
                'col': self._get_col
            },
            'set': {
                'element': lambda i, j: self._set_element(i, j, item),
                'row': lambda i: self._set_row(i, item),
                'col': lambda j: self._set_col(j, item)
            },
        }
        operation = operation_map[operation_type]

        row, col = self._unpack_index(index)
        row_intlike, col_intlike = isintlike(row), isintlike(col)

        if row_intlike:
            if col_intlike:
                return operation['element'](row, col)
            elif col == slice(None, None, None):
                return operation['row'](row)
            else:
                raise NotImplementedError(
                    'column index can either be int or empty')
        elif col_intlike:
            if row == slice(None, None, None):
                return operation['col'](col)
            else:
                raise NotImplementedError(
                    'row index can either be int or empty')
        else:
            raise NotImplementedError(
                'advanced indexing is not implemented yet')

    @abstractmethod
    def _get_row(self, index):
        raise NotImplementedError

    @abstractmethod
    def _set_col(self, index, item):
        raise NotImplementedError

    @abstractmethod
    def _set_element(self, i, j, item):
        raise NotImplementedError

    @abstractmethod
    def _set_row(self, index, item):
        raise NotImplementedError

    def __getitem__(self, index):
        self._get_or_set_item('get', index)

    def __setitem__(self, index, item):
        self._get_or_set_item('set', index, item)
