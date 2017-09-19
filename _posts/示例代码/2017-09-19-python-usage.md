---
layout: post
title: python usage
category: 示例代码
tags: code
keywords: python代码
description: 
---

# python reshape和matlab reshape的区别

```
mat = [1:12]

>>mat =

     1     2     3     4     5     6     7     8     9    10    11    12

reshape(mat,[3,4])

>>ans =

     1     4     7    10
     2     5     8    11
     3     6     9    12
```

```
mat = np.arange(1,13)

mat

>>array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])

r = np.reshape(mat,(3,4))

>>array([[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12]])

r.shape

>>(3, 4)
```

```
r = np.reshape(mat, (3,4), order="F")

r
array([[ 1,  4,  7, 10],
       [ 2,  5,  8, 11],
       [ 3,  6,  9, 12]])
```

需要在python程序中指明使用Fortran order, 如

```
np.reshape(matrix, (n,n), order="F")
```

Numpy默认是C order, Matlab是 Fortran order,就是python的reshape取行元素填充行元素，matlab取列元素填充列元素


