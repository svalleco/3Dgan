def safe_mkdir(path):
    '''
    Safe mkdir (i.e., don't create if already exists,
    and no violation of race conditions)
    '''
    from os import makedirs
    from errno import EEXIST
    try:
        makedirs(path)
    except OSError as exception:
        if exception.errno != EEXIST:
              raise exception
