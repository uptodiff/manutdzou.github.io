---
layout: post
title: python小程序
category: 示例代码
tags: code
keywords: python代码
description: 
---

# python连续帧图片写视频

```python
import cv2,os
dictionary='show'
size = (1280,672)
fps=20
video=cv2.VideoWriter('demo_show.avi', cv2.cv.CV_FOURCC('M','J','P','G'), fps,size)
for i in range(5782):
    name = str(i)+'.jpg'
    im_path = os.path.join('/home/dlg',dictionary,name)
    im = cv2.imread(im_path)
    video.write(im)
print 'done'
```

# python 用opencv检测视频中人脸

```python
import cv2
import os
import sys
 
OUTPUT_DIR = './my_faces'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
face_haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
count = 0
while True:
    print(count)
    if count < 10000:
        _, img = cam.read()
	gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_haar.detectMultiScale(gray_image, 1.3, 5)
	for face_x,face_y,face_w,face_h in faces:
	    face = img[face_y:face_y+face_h, face_x:face_x+face_w]
	    face = cv2.resize(face, (64, 64))
	    cv2.imshow('img', face)
	    cv2.imwrite(os.path.join(OUTPUT_DIR, str(count)+'.jpg'), face)
	    count += 1
	key = cv2.waitKey(30) & 0xff
	if key == 27:
            break
    else:
        break
```

# 用python requests模块下载数据

```python
import requests
import tarfile
 
url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
save_path = "my_path"
if not os.path.exists(save_path):
	os.makedirs(save_path)
 
filename = "save_file"
filepath = os.path.join(save_path, filename)
# 下载
if not os.path.exists(filepath):
	print("downloading...", filename)
	r = requests.get(url)
	with open(filepath, 'wb') as f:
	    f.write(r.content)
# 解压
tarfile.open(filepath, 'r:gz').extractall(filepath)
```
