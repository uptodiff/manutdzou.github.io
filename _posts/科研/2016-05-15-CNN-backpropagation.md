---
layout: post
title: CNN反向传播
category: 科研
tags: 深度学习
keywords: CNN反向传播理论
description: 
---

# CNN反向求导及推导

首先我们来看看CNN系统的目标函数，设有样本$\left({x_i},{y_i} \right)$共$m$个，CNN网络共有L层，中间包含若干个卷积层和pooling层，最后一层的输出为$f\left( {x_i} \right)$，则系统的loss表达式为(对权值进行了惩罚，一般分类都采用交叉熵形式)：

$$Loss =  - \frac{1}{m}\sum\nolimits_{i = 1}^m {y_i^'} \log f\left( {x_i} \right) + \lambda \sum\nolimits_{k = 1}^L {sum\left( {{{\left\| {{W_k}} \right\|}^2}} \right)}$$

## 问题一：求输出层的误差敏感项。

现在只考虑个一个输入样本$\left( {x,y} \right)$的情形，loss函数和上面的公式类似是用交叉熵来表示的，暂时不考虑权值规则项，样本标签采用one-hot编码，CNN网络的最后一层采用softmax全连接(多分类时输出层一般用softmax)，样本$\left( {x,y} \right)$经过CNN网络后的最终的输出用$f\left( {{x}} \right)$表示，则对应该样本的loss值为:

![1](/public/img/posts/CNN反向传播/1.png)

其中$f\left( {x} \right)$的下标$c$的含义见公式：
$$f{\left( x \right)_c} = p\left( {y = c|x} \right)$$

因为$x$通过CNN后得到的输出$f\left( {x} \right)$是一个向量，该向量的元素值都是概率值，分别代表着$x$被分到各个类中的概率，而$f\left( {x} \right)$中下标c的意思就是输出向量中取对应$c$那个类的概率值。
采用上面的符号，可以求得此时loss值对输出层的误差敏感性表达式为:

![2](/public/img/posts/CNN反向传播/2.png)

其中$e\left( y \right)$表示的是样本$x$标签值的one-hot表示，其中只有一个元素为1,其它都为0.
其推导过程如下（先求出对输出层某个节点c的误差敏感性，参考Larochelle关于DL的课件:Output layer gradient）,求出输出层中c节点的误差敏感值：

![3](/public/img/posts/CNN反向传播/3.png)