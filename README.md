Reimplementation of [Ranger-Mish](https://github.com/lessw2020/Ranger-Mish-ImageWoof-5) on pure PyTorch.

To reproduce the results, run
```python3 main.py --dataroot /path/to/imagewoof --num-classes 10 -sa -sym --lookahead --optim ralamb --flat 3.6 --sched cosine --seed 42```