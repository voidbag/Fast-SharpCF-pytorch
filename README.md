# Fast-SharpCF-pytorch
## Fast Implementation of Adversarial Collaborative Filtering for Free (RecSys23)

Fast Version of SharpCF (RecSys23).
Slow version of Pre-Post Processing is based on [BPR-pytorch](https://github.com/guoyang9/BPR-pytorch). \
I assume that the paper used pre/pos precessing of [BPR-pytorch](https://github.com/guoyang9/BPR-pytorch), because the Training Time(MovieLens-1M, 17.1s) is similar to [BPR-pytorch](https://github.com/guoyang9/BPR-pytorch).


### Contribution Points (Running Time Improvement: **11.8663x**)
1. Remove DataLoader used in [BPR-pytorch](https://github.com/guoyang9/BPR-pytorch) (Improvement: **46.4108x**)
    - Remove IPC(Inter Process Communiation) Overhead that Multi-Process DataLoader has
    - Remove Data Loading Process(CPU->GPU) Every Epoch
2. Negative Sampling (Improvement: **429.898x**)
    - Negative Sampling at Every Epoch is Done within only GPU, not CPU
3. HR, nDCG Calculation (Improvement: **125.357x**)
    - HR and nDCG is done within only GPU, not CPU

|                               |    01.train() |   02.neg_sample |   03.get_batch |   04.zero_grad |   05.backward |   05.forward |    06.step |     07.eval() |   08.hr_ndcg |    total |
|:------------------------------|--------------:|----------------:|---------------:|---------------:|--------------:|-------------:|-----------:|--------------:|-------------:|---------:|
| slow(origin)                  |   3.49839e-05 |      14.3219    |      2.68599   |      0.0379853 |      0.976263 |     0.37763  |   0.177941 |   0.000172035 |    2.54794   | 21.1258  |
| fast(optimized)               |   3.73626e-05 |       0.0333146 |      0.0578742 |      0.0405509 |      1.00162  |     0.424448 |   0.20211  |   3.7194e-05  |    0.0203255 |  1.78032 |
| improvements(fast over slow)  |   0.936335    |     429.898     |     46.4108    |      0.936733  |      0.974684 |     0.889698 |   0.880417 |   4.62535     |  125.357     | 11.8663  |
| paper                         | nan           |     nan         |    nan         |    nan         |    nan        |   nan        | nan        | nan           |  nan         | 17.1     |
| improvements(fast over paper) | nan           |     nan         |    nan         |    nan         |    nan        |   nan        | nan        | nan           |  nan         |  9.60503 |

Hardward Specs
    - CPU: Virtual Machine
    - GPU: RTX-3090-ti

## Latency Breakdown (Optimized Version vs Slow Version)
![newplot (1)](https://github.com/user-attachments/assets/13fa7d7a-2b72-4427-84f9-fad3f5c11561)


## nDCG over Epochs (Trajactory Loss is used after 40-th epoch, nDCG@10)
![newplot (2)](https://github.com/user-attachments/assets/e2131042-d9a5-44c3-9632-dc595980995e)


## Data
- Dataset is used in https://github.com/hexiangnan/adversarial_personalized_ranking/tree/master/Data
- To Select Sepecific Dataset('ml-1m', 'pinterrest-20', 'yelp'), modify dataset variable in config.py

## Prepare Dependencies and Dataset
```
git submodule sync
git submodule update --init --recursive
pip install -r requiremetns.txt
python ./download_dataset.py
```

## Reproduce Results in README.md

### Fast Version of SharpCF (warm epoch: 40)
```
python ./main.py --factor_num=128 --batch_size 4096 --epochs 1000 --free_warm 40 --free_lambda 4096 --lr 0.01 --top_k 10 --chk_period 100 --out_dir ./out_report_post
```

### Fast Version of BPR (warm epoch: 1000)
```
python ./main.py --factor_num=128 --batch_size 4096 --epochs 1000 --free_warm 1000 --free_lambda 4096 --lr 0.01 --top_k 10 --chk_period 100 --out_dir ./out_report_post
```

### Slow Version of SharpCF (warm epoch: 40) which Might Seem like Paper's Implementation.
```
python ./slow_main.py --factor_num=128 --batch_size 4096 --epochs 1000 --free_warm 40 --free_lambda 4096 --lr 0.01 --top_k 10 --chk_period 100 --out_dir ./out_slow_report_post
```

### Generate Plotly Plot (*.html) in out_dir
```
python ./generate_plot.py --sharp_dir ./out_report_post/batch_4096_lr_0.01_epochs_1000_freewarm_40_freelambda_4096.0/ --bpr_dir ./out_report_post/batch_4096_lr_0.01_epochs_1000_freewarm_1000_freelambda_4096.0/ --slow_dir ./out_slow_report_post/batch_4096_lr_0.01_epochs_1000_freewarm_40_freelambda_4096.0/ --out_dir ./out_plot
```


