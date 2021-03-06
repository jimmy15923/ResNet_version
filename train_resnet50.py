from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.applications.resnet50 import preprocess_input, ResNet50

import argparse
import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='ResNet version of V1/V2 and different normalization')

parser.add_argument('--keras', required=False, default=True)
parser.add_argument('--name', required=False, default=True)
parser.add_argument('--batch_size', required=True, default=32, type=int)
parser.add_argument('--norm', required=True, default="bn")
parser.add_argument('--epochs', required=False, default=100, type=int)
parser.add_argument("--optimizer", required=False, default="adam")
parser.add_argument("--gpu_id", required=False, default="1")
parser.add_argument("--gpu_fraction", required=False, default=0.3, type=float)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = args.gpu_fraction  #占用30%显存
sess = tf.Session(config=config)


use_keras = args.keras

# 資料路徑
DATASET_PATH = 'sample'

# 影像大小
IMAGE_SIZE = (224, 224)

# 影像類別數
NUM_CLASSES = 2

# 若 GPU 記憶體不足，可調降 batch size 或凍結更多層網路
BATCH_SIZE = args.batch_size

# 凍結網路層數
FREEZE_LAYERS = 0

# Epoch 數
NUM_EPOCHS = args.epochs


# 透過 data augmentation 產生訓練與驗證用的影像資料
train_datagen = ImageDataGenerator(rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   channel_shift_range=10,
                                   horizontal_flip=True,
                                   fill_mode='nearest',
                                   preprocessing_function=preprocess_input)
train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/valid',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE*4)

# 輸出各類別的索引值
for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))

# 以訓練好的 ResNet50 為基礎來建立模型，

if use_keras == "v0":
    print("Keras.application.ResNet: V1")
    # 捨棄 ResNet50 頂層的 fully connected layers
    net = ResNet50(include_top=False, weights=None, input_tensor=None,
                   input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
    x = net.output
    x = GlobalAveragePooling2D()(x)

    # 增加 Dense layer，以 softmax 產生個類別的機率值
    output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)
    net_final = Model(inputs=net.input, outputs=output_layer)
    
    if args.optimizer == "adam":
        print("Optimizer: Adam")
        net_final.compile(optimizer=Adam(lr=1e-4),
                          loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        print("Optimizer: SGD")
        net_final.compile(optimizer=SGD(
            lr=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

elif use_keras == "v1":
    print("ResNet: V1")
    from resnet_graph import *
    input_image = KL.Input(shape=[224, 224, 3], name="input_image")
    _, C2, C3, C4, C5 = resnet_graph(input_image, 'resnet50', stage5=True, norm_use=args.norm)

    gap = KL.GlobalAveragePooling2D()(C5)
    output = KL.Dense(2, activation="softmax")(gap)

    net_final = KM.Model(inputs=input_image, outputs=output)

    if args.optimizer == "adam":
        print("Optimizer: Adam")
        net_final.compile(optimizer=Adam(lr=1e-4),
                          loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        print("Optimizer: SGD")
        net_final.compile(optimizer=SGD(
            lr=0.01, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])

else:
    print("ResNet: V2")
    from resnetv2_graph import *

    pretrain_modules = ResNet50V2(include_top=False, input_shape=(224,224,3), norm_use=args.norm, weights=None)
    gap = tf.keras.layers.GlobalAveragePooling2D()(pretrain_modules.output)
    logit = tf.keras.layers.Dense(units=2, name="logit")(gap)
    out = tf.keras.layers.Activation("softmax", name="output")(logit)

    net_final = tf.keras.Model(inputs=[pretrain_modules.input], outputs=[out])
    
    if args.optimizer == "adam":
        print("Optimizer: Adam")
        net_final.compile(optimizer=Adam(lr=1e-4),
                          loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        print("Optimizer: SGD")
        net_final.compile(optimizer=SGD(
            lr=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# 輸出整個網路結構
print(net_final.summary())

from tensorflow.python.keras.callbacks import LearningRateScheduler
 
# learning rate schedule
def step_decay(epoch):
    lr = 0.01
    
    if (epoch > 30) & (epoch < 60):
        return lr /10
    elif (epoch > 60):
        return lr /100
    else:
        return lr
    
if args.optimizer == "adam":
    # 訓練模型
    history = net_final.fit_generator(train_batches,
                            steps_per_epoch = train_batches.samples // BATCH_SIZE,
                            validation_data = valid_batches,
                            validation_steps = valid_batches.samples // (BATCH_SIZE*4),
                            epochs = NUM_EPOCHS,
                            use_multiprocessing=True,
                            workers=4)
else:
     # 訓練模型
    lrate = LearningRateScheduler(step_decay, verbose=1)
    callbacks_list = lrate
    history = net_final.fit_generator(train_batches,
                            steps_per_epoch = train_batches.samples // BATCH_SIZE,
                            validation_data = valid_batches,
                            validation_steps = valid_batches.samples // (BATCH_SIZE*4),
                            epochs = NUM_EPOCHS,
                            use_multiprocessing=True,
                            workers=4,
                            callbacks=[callbacks_list]) 

df = pd.DataFrame(history.history)
df.to_csv("logs/{}.csv".format(args.name), index=False)
