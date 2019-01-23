# ResNet_version

This repo is to compare the performance of ResNet-50-V1 and ResNet-50-V2 on Kaggle Cat/Dog Classification. Data is located on 
DGX-staion: /home/jimmy15923/benchmarks/cat_dog/sample

ResNet graph is written in tf.keras, running experiments by

```
python train_resnet.py --keras=v1 --batch_size=32 --norm=bn --name=test

--keras: Speicy the version to use. keras / v1 / v2
--batch_size: Batch size
--epochs: Number of epochs
--norm: Which normalization to use. bn / gn / sn
--name: Name of the learning curve plot
```

---
## Experiments

### We compare the performance with different version of ResNet-50 trained from scratch. 
The result shows that loss of v2 is unstable but more accurate than v1 

Hyper-parameter setting ares
- batch_size=32
- optimizer=Adam(lr=1e-4)
- batch normalization
- same augmentation
- epochs=100

**Keras application ResNet-50-V1**
![keras](v0_bn_32_result.png)

**Our own graph ResNet-50-V1**
![v1](v1_bn_32_result.png)

**Our own graph ResNet-50-V2**
![v2](v2_bn_32_result.png)

### We compare the performance with differenet normalization (BN/GN/SN) of same model.
The result shows that loss of SN is lower than other normalization but more bumpy.

Hyper-parameter setting ares
- batch_size=32
- optimizer=Adam(lr=1e-4)
- same augmentation
- epochs=100

**Batch Normalization**
![BN](v2_bn_32_result.png)

**Group Normalization**
![GN](v2_gn_32_result.png)

**Switch Normalization**
![SN](v2_sn_32_result.png)
