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