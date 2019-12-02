import os
import pickle
import shutil


class SerializedParams:
    def __init__(self, outpath=None):
        self.outpath = outpath
        self.param_names = []
        if outpath is not None:
            if os.path.exists(outpath):
                print('Clearing previous contents of {}'.format(outpath))
                shutil.rmtree(outpath, ignore_errors=True)
            os.mkdir(outpath)

    def __getitem__(self, name):
        return getattr(self, name)

    def register_param(self, name, value):
        assert name not in self.param_names
        self.param_names.append(name)
        setattr(self, name, value)

    def to_disc(self, epoch):
        dict = {}
        for param in self.param_names:
            dict[param] = getattr(self, param)
        with open(os.path.join(self.outpath, 'params_{}.pkl'.format(epoch)), 'wb') as fd:
            pickle.dump(dict, fd)

    @classmethod
    def from_disc(cls, in_fp):
        sm = SerializedParams()
        with open(in_fp, 'rb') as fd:
            dict = pickle.load(fd)
        for k, v in dict.items():
            sm.register_param(k, v)
        return sm
