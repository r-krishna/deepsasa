import numpy as np


# FROM TAPE https://github.com/songlab-cal/tape/blob/master/tape/utils/utils.py
class IncrementalNPZ(object):
    # Modified npz that allows incremental saving, from https://stackoverflow.com/questions/22712292/how-to-use-numpy-savez-in-a-loop-for-save-more-than-one-array  # noqa: E501
    def __init__(self, file):
        import tempfile
        import zipfile
        import os

        if isinstance(file, str):
            if not file.endswith('.npz'):
                file = file + '.npz'

        compression = zipfile.ZIP_STORED

        zipfile = self.zipfile_factory(file, mode="w", compression=compression)

        # Stage arrays in a temporary file on disk, before writing to zip.
        fd, tmpfile = tempfile.mkstemp(suffix='-numpy.npy')
        os.close(fd)

        self.tmpfile = tmpfile
        self.zip = zipfile
        self._i = 0

    def zipfile_factory(self, *args, **kwargs):
        import zipfile
        import sys
        if sys.version_info >= (2, 5):
            kwargs['allowZip64'] = True
        return zipfile.ZipFile(*args, **kwargs)

    def savez(self, *args, **kwds):
        import os
        import numpy.lib.format as fmt

        namedict = kwds
        for val in args:
            key = 'arr_%d' % self._i
            if key in namedict.keys():
                raise ValueError("Cannot use un-named variables and keyword %s" % key)
            namedict[key] = val
            self._i += 1

        try:
            for key, val in namedict.items():
                fname = key + '.npy'
                fid = open(self.tmpfile, 'wb')
                with open(self.tmpfile, 'wb') as fid:
                    fmt.write_array(fid, np.asanyarray(val), allow_pickle=True)
                self.zip.write(self.tmpfile, arcname=fname)
        finally:
            os.remove(self.tmpfile)

    def close(self):
        self.zip.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()