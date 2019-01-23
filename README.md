# ResNet_version

This repo is to compare the performance of ResNet-50-V1 and ResNet-50-V2 on Kaggle Cat/Dog Classification.

ResNet graph is written in tf.keras, running experiments by

```
python train_resnet.py --keras=v1 --batch_size=32 --norm=bn --name=test
```
--keras: Speicy the version to use. keras / v1 / v2
--batch_size: Batch size
--epochs: Number of epochs
--norm: Which normalization to use. bn / gn / sn
--name: Name of the learning curve plot
