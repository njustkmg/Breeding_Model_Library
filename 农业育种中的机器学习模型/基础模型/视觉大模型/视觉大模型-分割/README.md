## 视觉大模型--分割
---

### 目录
1. [概述](#概述)
4. [所需环境 Environment](#所需环境)
5. [文件下载 Download](#文件下载)
6. [训练步骤 How2train](#训练步骤)
7. [预测步骤 How2predict](#预测步骤)
8. [评估步骤 miou](#评估步骤)
9. [参考资料 Reference](#Reference)

## 概述

我们将视觉大模型作物表型分割数据集上进行了微调。

<img src="assets/image-20230506115555906.png" alt="image-20230506115555906" style="zoom:50%;" />

### 所需环境
torch==1.2.0

### 文件下载
在ImageNet上预训练的swin-transformer权重可以在百度云下载。     
链接: https://pan.baidu.com/s/18Ze7YMvM5GpbTlekYO8bcA     
提取码: 5wym   

在我们的作物表型分割上微调过的swin-transformer权重也可以在百度云下载。   
链接：https://pan.baidu.com/s/1d-7JsdgSF2u72BTqFrvmrg?pwd=g43t 
提取码：g43t

训练所用的VOC格式的作物分割迷你数据集也可以在百度云下载。   
链接：https://pan.baidu.com/s/1kuWon5NaDr_iWBT0UP0Olw?pwd=zivn  
提取码：zivn  

### 训练步骤
1、本文使用VOC格式进行训练。  
2、训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的SegmentationClass中。    
3、训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。    
4、在训练前利用voc_annotation.py文件生成对应的txt。    
5、在train.py文件夹下面，选择自己要使用的主干模型和下采样因子。本文提供的主干模型有mobilenet和xception。下采样因子可以在8和16中选择。需要注意的是，预训练模型需要和主干模型相对应。   
6、注意修改train.py的num_classes为分类个数+1。    
7、运行train.py即可开始训练。  

### 预测步骤
#### a、使用预训练权重
1、下载完库后解压，如果想用backbone为mobilenet的进行预测，直接运行predict.py就可以了；如果想要利用backbone为xception的进行预测，在百度网盘下载deeplab_xception.pth，放入model_data，修改deeplab.py的backbone和model_path之后再运行predict.py，输入。  
```python
img/1.jpg
```
可完成预测。       

#### b、使用自己训练的权重
1、按照训练步骤训练。    
2、在deeplab.py文件里面，在如下部分修改model_path、num_classes、backbone使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，num_classes代表要预测的类的数量加1，backbone是所使用的主干特征提取网络**。    
```python
_defaults = {
    #----------------------------------------#
    #   model_path指向logs文件夹下的权值文件
    #----------------------------------------#
    "model_path"        : 'model_data/swin_tiny_patch4_window7_224_phe.enotype_segmentation.pth',
    #----------------------------------------#
    #   所需要区分的类的个数+1
    #----------------------------------------#
    "num_classes"       : 6,
    #----------------------------------------#
    #   所使用的的主干网络
    #----------------------------------------#
    "backbone"          : "swin_transfomer_tiny",
    #----------------------------------------#
    #   输入图片的大小
    #----------------------------------------#
    "input_shape"       : [512, 512],
    #----------------------------------------#
    #   下采样的倍数，一般可选的为8和16
    #   与训练时设置的一样即可
    #----------------------------------------#
    "downsample_factor" : 16,
    #--------------------------------#
    #   blend参数用于控制是否
    #   让识别结果和原图混合
    #--------------------------------#
    "blend"             : True,
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    "cuda"              : True,
}
```
3、运行predict.py，输入    
```python
img/1.jpg
```
可完成预测。    

### 评估步骤
1、设置get_miou.py里面的num_classes为预测的类的数量加1。  
2、设置get_miou.py里面的name_classes为需要去区分的类别。  
3、运行get_miou.py即可获得miou大小。  

### Reference
https://github.com/ggyyzm/pytorch_segmentation  
https://github.com/bonlime/keras-deeplab-v3-plus  
https://github.com/bubbliiiing/classification-pytorch/tree/main  
https://github.com/bubbliiiing/deeplabv3-plus-pytorch