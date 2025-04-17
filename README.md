
# Basic Image Classification

PyTorch code for image classification using CNNs, along with some ablation experiments to improve learning without over-fitting

---

## Theory

Please note that the theory corresponding to the codes is hosted on [[link]]

## Python Environment Setup

Ensure the following is installed in your python environment:

- Numpy - required for vectorized operations
- Pytorch - required for automatic backpropagation over compute graphs (try installing the hardware accelerated version if possible)
- Torchvision - required for downloading the image datasets and pre-processing images when loading data

## Pytorch Neural networks

Execute the following script to train Pytorch Neural Network models for image classification:

```bash
python main.py <version_number>
```

|Version|Dataset|Train Acc|Test Acc|Params|Optimizer|Remarks|
|:-----:|:-----------:|:-------:|:------:|:----:|---------|----|
|1      |MNIST        |0.95     |0.95    |80K   |SGD      |We start with a simple MLP on MNIST dataset|
|2      |Fashion-MNIST|0.87     |0.85    |80K   |SGD      |Since MLP on MNIST is already performing well, we switch to FashionMNIST where there is room for improvement|
|3      |Fashion-MNIST|0.88     |0.87    |9K    |SGD      |We switch to a simple conv-net, which exploits spatial correlation in the data and gets more acc with fewer params|
|4      |Fashion-MNIST|0.90     |0.89    |1M    |SGD      |We created an over-powered model with an excessive number of params to try and overfit to the dataset, but that didn't work|
|5      |Fashion-MNIST|0.99     |0.92    |1M    |Adam     |Just changing the optimizer to Adam led to overfitting. But test results are also good, thus less room for improvement |
|6      |Cifar10      |0.75     |0.66    |33K   |Adam     |We switch to a simple conv-net on the Cifar10 dataset, but the model doesn't perform well even on the train set|
|7      |Cifar10      |0.87     |0.66    |33K   |Adam     |Previous model didn't train due to vanishing gradient. We use batch normalization to improve learning|
|8      |Cifar10      |0.99     |0.76    |0.6M  |Adam     |We switch to a bigger conv-net and overfit to the dataset|
|9      |Cifar10      |0.95     |0.81    |0.6M  |Adam     |Since we are overfitting, we use dropout regularization to improve test accuracy at the cost of some training accuracy|

After execution, you should get outputs like the following:

```bash
$ python main.py 9
Files already downloaded and verified
Files already downloaded and verified
Model Parameter Count: 655882
Training...
100%|███████████████████████████████| 782/782 [00:30<00:00, 25.69it/s]
Epoch 1 / 25 | Train Loss: 1.2303 | Train Acc: 0.5574
100%|███████████████████████████████| 782/782 [00:27<00:00, 28.30it/s]
Epoch 2 / 25 | Train Loss: 0.8746 | Train Acc: 0.6923
100%|███████████████████████████████| 782/782 [00:28<00:00, 27.91it/s]
Epoch 3 / 25 | Train Loss: 0.7241 | Train Acc: 0.7448
100%|███████████████████████████████| 782/782 [00:27<00:00, 28.24it/s]
Epoch 4 / 25 | Train Loss: 0.6334 | Train Acc: 0.7784
.
.
.
100%|███████████████████████████████| 782/782 [00:27<00:00, 28.31it/s]
Epoch 23 / 25 | Train Loss: 0.1519 | Train Acc: 0.9480
100%|███████████████████████████████| 782/782 [00:27<00:00, 28.25it/s]
Epoch 24 / 25 | Train Loss: 0.1412 | Train Acc: 0.9501
100%|███████████████████████████████| 782/782 [00:27<00:00, 28.16it/s]
Epoch 25 / 25 | Train Loss: 0.1364 | Train Acc: 0.9527
Training complete!
Testing: 100%|██████████████████████| 157/157 [00:21<00:00,  7.38it/s]
Test Loss: 0.8168 | Test Acc: 0.8058
```
# Basic-Image-Classification-ML
