Reimplementation of [Ranger-Mish](https://github.com/lessw2020/Ranger-Mish-ImageWoof-5) on pure PyTorch.

To reproduce the results, run
```python3 main.py --dataroot /path/to/imagewoof --num-classes 10 -sa -sym --lookahead --optim ralamb --flat 3.6 --sched cosine --seed 42```

I also ran it on ImageNet, for the same settings (i.e., 128x128 input size for 5 epochs). 
As per ["Fixing the train-test resolution discrepancy"](https://arxiv.org/abs/1906.06423), 
ResNet-50 achieves 73.3% top-1 accuracy if both trained and validated on 128x128 inputs.
After 5 epochs (100095 batches), which took 7:22 hours, 
MXResNet-50 achieved 66.02% top-1 accuracy.