import numpy as np

from .dataset import Dataset


class DataSuperset(Dataset):
    """
    Superset of datasets.
    """
    def __init__(self, tag=''):
        self.tag = tag
        self.datasets = list()
        self.p = None

    def __call__(self, spec=None):
        return self.random_sample(spec=spec)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += self.tag
        format_string += ')'
        return format_string

    def sanity_check(self, spec):
        sanity = True
        for dset in self.datasets:
            sanity &= dset.sanity_check(spec)
        return sanity

    def add_dataset(self, dset):
        assert isinstance(dset, Dataset)
        self.datasets.append(dset)

    def random_sample(self, spec=None, idx=None):
        dset = self.random_dataset() if idx is None else self.datasets[idx]
        return dset(spec=spec)

    def set_sampling_weights(self, p=None):
        if p is None:
            p = [d.num_samples() for d in self.datasets]
        p = np.asarray(p, dtype='float32')
        p = p/np.sum(p)
        assert len(p)==len(self.datasets)
        self.p = p

    def random_dataset_idx(self):
        assert len(self.datasets) > 0
        if self.p is None:
            self.set_sampling_weights()
        return np.random.choice(len(self.datasets), size=1, p=self.p)[0]

    def random_dataset(self):
        return self.datasets[self.random_dataset_idx()]

    def num_samples(self, spec=None):
        return sum([dset.num_samples(spec=spec) for dset in self.datasets])
