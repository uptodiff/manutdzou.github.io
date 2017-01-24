---
layout: post
title: Caffe Ubuntu下环境配置
category: 科研
tags: 深度学习
keywords: 系统环境
description: 
---

# Caffe 安装配置步骤:

1，安装开发所需的依赖包

```Shell
sudo apt-get install build-essential  # basic requirement
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler #required by caffe
```

2，安装CUDA 7.0

离线.deb安装，官网下载deb文件，切换到下载的deb所在目录，执行下边的命令

```Shell
sudo dpkg -i cuda-repo-<distro>_<version>_<architecture>.deb
sudo apt-get update 
sudo apt-get install cuda
```

然后重启电脑：

```Shell
sudo reboot
```

3，安装cuDNN

下载cudnn-7.0-linux-x64-v3.0-rc.tgz

```shell
tar -zxvf cudnn-7.0-linux-x64-v3.0-rc.tgz
cd cudnn-7.0-linux-x64-v3.0-rc  (cuda目录)
sudo cp lib* /usr/local/cuda/lib64/  (切换到lib64目录)
sudo cp cudnn.h /usr/local/cuda/include/  （切换到include目录）
```

更新软连接

```Shell
cd /usr/local/cuda/lib64/
sudo rm -rf libcudnn.so libcudnn.so.7.0
sudo ln -s libcudnn.so.7.0.58 libcudnn.so.7.0
sudo ln -s libcudnn.so.7.0 libcudnn.so
```

4，设置环境变量

在/etc/profile中添加CUDA环境变量

```Shell
sudo gedit /etc/profile
```

在文件后面加入PATH=/usr/local/cuda/bin:$PATH export PATH 保存后, 执行下列命令, 使环境变量立即生效

```Shell
source /etc/profile
```

同时需要添加lib库路径： 在 /etc/ld.so.conf.d/加入文件 cuda.conf, 过程如下

```Shell
vi /etc/ld.so.conf.d/cuda.conf
```

写入/usr/local/cuda/lib64打开编辑器，按i进入插入模式写入/usr/local/cuda/lib64然后：wq保存,执行下列命令使之立刻生效

```Shell
sudo ldconfig
```

5，安装CUDA SAMPLE

进入/usr/local/cuda/samples, 执行下列命令来build samples

```Shell
sudo make all –j8
```

整个过程大概10分钟左右, 全部编译完成后， 进入 samples/bin/x86_64/linux/release, 运行deviceQuery

```Shell
./deviceQuery
```

如果出现显卡信息， 则驱动及显卡安装成功

6，安装Intel MKL 或Atlas

安装命令

```Shell
sudo apt-get install libatlas-base-dev
```

7，安装OpenCV

- 下载安装脚本
- 进入目录 Install-OpenCV/Ubuntu/2.4
- 执行脚本

```Shell
sh sudo ./opencv2_4_10.sh
```

8，安装Caffe所需要的Python环境

网上介绍用现有的anaconda，我反正不建议，因为路径设置麻烦，很容易出错，而且自己安装很简单也挺快的。
首先需要安装pip

```Shell
sudo apt-get install python-pip
```

再下载caffe，我把caffe放在用户目录下，安装git

```Shell
sudo apt-get install git
```

cd回到home目录

```Shell
git clone https://github.com/BVLC/caffe.git （注意版本，可能获得版本存在问题，可以直接去caffe社区下载最新版本）
```

再转到caffe的python目录，安装scipy

```Shell
cd caffe/python
sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
```

最后安装requirement里面的包，需要root权限

```Shell
sudo su
for req in $(cat requirements.txt); do pip install $req; done
```

如果提示报错，一般是缺少必须的包引起的，直接根据提示 pip install <package-name>就行了。
安装完后退出root权限

```Shell
exit 
```

9，编译Caffe

终于来到这里了

进入caffe-master目录，复制一份Makefile.config.examples

```Shell
cp Makefile.config.example Makefile.config
```

修改其中的一些路径，保存退出

编译

```Shell
make all -j4
make test
make runtest
```

10，编译Python wrapper

```Shell
make  pycaffe
```

11，安装 matlab

降级安装gcc4.7  g++4.7 因为matlab编译只支持到4.7

```Shell
sudo apt-get install –y gcc-4.7   sudo apt-get install –y g++-4.7
cd /usr/bin
sudo rm gcc   sudo ln –s gcc-4.7 gcc
sudo rm g++   sudo ln –s g++-4.7 g++
gcc –version 查看版本号
```

编译Matlab wrapper

```Shell
make matcaffe 
```
