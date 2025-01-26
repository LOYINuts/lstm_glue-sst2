# 项目介绍

使用pytorch的LSTM简单实现情感二分类的一个项目

数据集在data文件夹下，本项目是基于thunlp的大模型课程的第二章的exercise而来，但是源代码并没有使用pytorch的dataset和dataloader，以至于很多操作略显繁琐以及不能学习到pytorch的常规处理。所以本人完成了这个作业，也是记录一下自己的学习过程

# 文件介绍

main.py 是训练文件的程序的入口，注释什么的应该都比较完备。

Datasetutils.py 是dataset，词典等相关设置

mymodel.py 是网络模型的文件

train.py 是训练函数的文件

predict.py 是简单的写的评估模型的代码

# 依赖相关

Python 版本：3.9
相关包依赖已经在requirements.txt文件中。