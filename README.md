# Exploration-on-Batch-Normalization
In this PJ I build a simple Conv2d network and implement some beneficial modifications based on it. Also I conduct some investigation on the network architecture and training process, ultimately my best model achieves a 96.68% test accuracy and CrossEntropy test loss reduced to
0.0012. 
The explorations done are as follows:
1. Increasing batch size;
2. Implementing dropout;
3. Implementing Residual Connection;
4. Try different number of neurons/filters;
5. Try different loss functions;
6. Try different Activation Function;
7. Try different optimizers using torch.optim;
8. Network interpretation

## different resnet
![img](https://github.com/Connor-Shen/Exploration-on-CIFAR10-and-BN/blob/main/img/Resnet.png)
## dropout
![img](https://github.com/Connor-Shen/Exploration-on-CIFAR10-and-BN/blob/main/img/drop_out.png)
## loss functions
![img](https://github.com/Connor-Shen/Exploration-on-CIFAR10-and-BN/blob/main/img/loss_type.png)
## Activation Function
![img](https://github.com/Connor-Shen/Exploration-on-CIFAR10-and-BN/blob/main/img/activation_function.png)
## Optimizer
![img](https://github.com/Connor-Shen/Exploration-on-CIFAR10-and-BN/blob/main/img/optimizer_type.png)

# Batch-Normalization (BN) 
Batch-Normalization (BN) is an algorithmic method which makes the training of Deep Neural Networks
(DNN) faster and more stable. It consists of normalizing activation vectors from hidden layers using the first and the second statistical
moments (mean and variance) of the current batch. This normalization step is applied right before (or right after) the nonlinear function.
Here I mainly compared VGG-A with and without BN and drew the Loss Landscape.

## VGG with Batch-Normalization
![img](https://github.com/Connor-Shen/Exploration-on-CIFAR10-and-BN/blob/main/img/VGG_type.png)
## loss landscape
![img](https://github.com/Connor-Shen/Exploration-on-CIFAR10-and-BN/blob/main/img/loss_landscape.png)

