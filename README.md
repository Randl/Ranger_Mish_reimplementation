Reimplementation of [Ranger-Mish](https://github.com/lessw2020/Ranger-Mish-ImageWoof-5) on pure PyTorch.

To reproduce the results, run

```python3 main.py --dataroot /path/to/imagewoof --num-classes 10 -sa -sym --lookahead --optim ralamb --flat 3.6 --sched cosine --seed 42```

I also ran it on ImageNet, for the same settings (i.e., 128x128 input size for 5 epochs). 
As per ["Fixing the train-test resolution discrepancy"](https://arxiv.org/abs/1906.06423), 
ResNet-50 achieves 73.3% top-1 accuracy if both trained and validated on 128x128 inputs.
After 5 epochs (100095 batches), which took 7:22 hours, 
MXResNet-50 achieved 66.02% top-1 accuracy.

## BlurPool
This part is based on paper ["Making Convolutional Networks Shift-Invariant Again"](https://richzhang.github.io/antialiased-cnns/) and is due [Dmytro Mishkin](https://twitter.com/ducha_aiki/status/1216859128397750278).

However the scheme is a bit different from the proposed in paper:

|Operation|Proposed in paper| Used|
|---|---|---|
|Convolution with stride | Convolution without stride + Blurred downsample | Convolution with stride|
|Max pooling | Max pooling without stride + Blurred downsample | Max pooling without stride + Blurred downsample|
|Avg pooling | Blurred downsample | Max pooling without stride + Blurred downsample|
The orginal scheme, either full or partial seems to work worse.

Regular MXResNet-50 gives
for seeds [1,2,3,4,5] the following accuracies: [0.752,0.746,0.762,0.764,0.752], i.e., the accuracy is 75.52±0.67%

BlurPool for same seeds gives [0.776,0.79,0.762,0.768,0.756], i.e., 77.04±1.18%