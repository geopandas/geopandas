cimport numpy as np

cdef _cy_series_op_fast(np.ndarray[object, ndim=1] array, object geometry, str op)
cdef _cy_series_op_fast_unprepared(np.ndarray[object, ndim=1] array, object geometry, str op)
cdef _cy_series_op_slow(object this, object other, str op, kwargs)
cdef _py_geo_unary_op(np.ndarray[object, ndim=1] this, str op)
cdef _cy_geo_unary_op(np.ndarray[object, ndim=1] array, str op)
