---
layout: post
title: tensorflow使用记录
category: tensorflow
tags: 深度学习
keywords: tf学习
description: tf学习
---

# 减均值后图像复原

```
image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
channel_mean = tf.constant(np.array([123.68,116.779,103.938], dtype=np.float32))
image_before_process = tf.add(image,channel_mean)
```

# 在构造图时候查看tensor的shape

```
tensor.get_shape()
```

# tf.app.run()pudb调试时候遇到参数错误，运行时候正常

使用tf.app.run(main=main)

# tensorflow查看可用设备

```
from tensorflow.python.client import device_lib as _device_lib
print _device_lib.list_local_devices()
```

# 从checkpoint中读取tensor

```
import os
from tensorflow.python import pywrap_tensorflow

checkpoint_path = os.path.join(model_dir, "model.ckpt")
# Read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
# Print tensor name and values
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key))
```

或者

```
import tensorflow as tf
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
ckpt = tf.train.get_checkpoint_state('./model_dir/')                          # 通过检查点文件锁定最新的模型
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')   # 载入图结构，保存在.meta文件中
var = [v for v in tf.trainable_variables()]

with tf.Session() as sess:
    saver.restore(sess,ckpt.model_checkpoint_path)
    logging.info("load parameter done")
    parameter = []
    for i in range(len(var)):
        parameter.append(sess.run(var[i]))
		#logging.info(var[i].name)
        logging.info(var[i].initialized_value())
```