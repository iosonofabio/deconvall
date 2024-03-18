def _infer_dtype(dtype):
    if str(dtype) == 'object':
        return 'S'
    if str(dtype).startswith('S'):
        return 'S'
    if str(dtype).startswith('U'):
        return 'S'
    # FIXME: improve this
    if str(dtype) == 'category':
        return 'S'

    return dtype

