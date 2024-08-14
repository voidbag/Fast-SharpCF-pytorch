import torch
import numpy as np

from BPRpytorch.data_utils import load_all
from BPRpytorch.data_utils import BPRData

# HW Acceleration
class FastBPRData(BPRData):
    def __init__(self, features, num_item, train_mat=None, num_ng=0, is_training=None, reuse_order=False):
        super().__init__(features, num_item, train_mat, num_ng, is_training)
        self.order = None
        self.reuse_order = reuse_order
        self.train_mat = self.train_mat.toarray() 
        self.features = torch.tensor(self.features, dtype=torch.int32).cuda()
        if self.is_training:
            tensor_mat = torch.tensor(self.train_mat).cuda()
            self.arg_sorted = tensor_mat.argsort(dim=1, descending=False, stable=True) # HW accelration
            self.tensor_uid = self.features[:,0].cuda().repeat_interleave(self.num_ng)
            self.tensor_pos = self.features[:,1].cuda().repeat_interleave(self.num_ng)

            num_samples = self.train_mat.sum(axis=1) # number of filled items per uid
            num_empty = self.train_mat.shape[1] - num_samples # number of unfilled items per uid
            self.tensor_uid_max_neg = torch.tensor(num_empty, dtype=torch.int32).cuda()[self.tensor_uid] 
            self.tensor_train = None

    def ng_sample(self, saved_order=None):
        assert self.is_training, "no need to do negative sampling when testing"
        int_max = int(torch.iinfo(torch.int64).max)
        tensor_neg = torch.randint(int_max, self.tensor_uid.size()).cuda() % self.tensor_uid_max_neg
        tensor_neg = self.arg_sorted[self.tensor_uid, tensor_neg]

        self.tensor_train = torch.cat([self.tensor_uid.view(-1, 1), self.tensor_pos.view(-1, 1), tensor_neg.view(-1, 1)], dim=1)
        
        if self.reuse_order:
            if saved_order is not None:
                self.order = saved_order
            elif self.order is None:
                self.order = torch.randperm(self.tensor_train.size()[0])
        else:
            assert saved_order is None, "use saved_order only with reuse_order!"
            self.order = torch.randperm(self.tensor_train.size()[0])

        self.tensor_train = self.tensor_train[self.order, :] #full shuffle

    def __len__(self):
        return self.num_ng * len(self.features) if self.is_training else len(self.features)

    def __getitem__(self, idx):
        if self.is_training:
            return self.tensor_train[idx, :]

        return self.features[idx, 0], self.features[idx, 1], self.features[idx, 1]
