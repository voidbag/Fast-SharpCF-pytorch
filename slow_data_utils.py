import numpy as np

from BPRpytorch.data_utils import load_all
from BPRpytorch.data_utils import BPRData

class SlowBPRData(BPRData):
    def __init__(self, features, num_item, train_mat=None, num_ng=0, is_training=None, reuse_order=False):
        super().__init__(features, num_item, train_mat, num_ng, is_training)
        self.reuse_order = reuse_order
        self.order = None

    def ng_sample(self, saved_order=None):
        super().ng_sample()
        assert self.is_training, "no need to do negative sampling when testing"
        order = None
        if self.reuse_order:
            if saved_order is not None:
                self.order = saved_order
            elif self.order is None:
                self.order = np.random.permutation(len(self))
        else:
            assert saved_order is None, "use saved_order only with reuse_order!"
            self.order = np.ramdom.permutation(len(self))

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        if self.order is not None:
            idx = self.order[idx] #relocation
        return super().__getitem__(idx)
