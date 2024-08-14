import numpy as np
import torch

def metrics(model, test_dataset, top_k, num_unit, idx_gt=0, repeats=1000):
    torch_gt = torch.tensor((idx_gt,), dtype=torch.int32).cuda()
    batch_size_test = (num_unit) * repeats
    num_batch_test = (len(test_dataset) + (batch_size_test-1)) // (batch_size_test)
    num_rows = len(test_dataset) // num_unit
    assert num_rows * num_unit == len(test_dataset)
    hr = 0
    ndcg = 0
    li_hr = list()
    li_ndcg = list()
    for idx_batch_test in range(num_batch_test):
        start = idx_batch_test * batch_size_test
        end = min(start + batch_size_test, len(test_dataset))
        user, item_i, item_j = test_dataset[start:end]
        pred_i, _ = model(user, item_i, item_j)
        pred_i = pred_i.view(-1, num_unit)
        idx_sorted = torch.argsort(pred_i, dim=1, descending=True)
         
        has_true = idx_sorted[:, :top_k] == torch_gt
        if pred_i.isnan().any().item():
            has_true[:] = False
        _hr = (has_true).sum(dim=1).type(torch.float32).cpu().numpy()
        _ndcg = (has_true / torch.log2(torch.arange(2, top_k + 2).tile((idx_sorted.shape[0], 1))).cuda()).sum(dim=1).cpu().numpy()
        li_hr.append(_hr)
        li_ndcg.append(_ndcg)
    
    hr = np.concatenate(li_hr, axis=0)
    ndcg = np.concatenate(li_ndcg, axis=0)
    assert hr.shape[0] == num_rows
    assert ndcg.shape[0] == num_rows
    return hr, ndcg
