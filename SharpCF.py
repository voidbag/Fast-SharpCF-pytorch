import torch
import torch.nn as nn
from BPRpytorch.model import BPR

class SharpCF(BPR):
    def __init__(self, user_num, item_num, factor_num, win_shape, ds_order):
        super().__init__(user_num, item_num, factor_num)
        self.win_shape = win_shape
        self.window = nn.Parameter(torch.zeros(win_shape, dtype=self.embed_user.weight.dtype), requires_grad=False)
        self.ds_order = nn.Parameter(ds_order, requires_grad=False)

    def cuda(self):
        super().cuda()
        self.window = self.window.cuda()
        if self.ds_order is not None:
            self.ds_order = self.ds_order.cuda()

    def cpu(self):
        super().cpu()
        self.window = self.window.cpu()
        if self.ds_order is not None:
            self.ds_order = self.ds_order.cpu()

    def forward(self, user, item_i, item_j):
        pred_i, pred_j = super().forward(user, item_i, item_j)
        return pred_i, pred_j

    def update_order(self, order):
        self.ds_order = nn.Parameter(torch.clone(order).to(self.ds_order.device), requires_grad=False)

