# -*- coding: UTF-8 -*-

import os
import gzip


def read_gz_file(path):
    if os.path.exists(path):
        with gzip.open(path, 'rb') as pf:
            first_line = pf.readline().strip().decode('utf-8').replace('\x00', '').replace('\n', '')
            valid_info = first_line.split()[-4:]
            yield ' '.join(valid_info)
            for line in pf:
                res = line.strip().decode('utf-8').replace('\x00', '').replace('\n', '')
                if len(res) > 0:
                    yield res
    else:
        raise IOError('  The path [{}] is not exist!'.format(path))
