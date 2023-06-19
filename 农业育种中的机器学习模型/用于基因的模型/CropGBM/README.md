# 欢迎使用 CropGBM！

## 介绍

CropGBM是一个多功能Python 程序，集成了数据预处理、种群结构分析、SNP选择、表型预测和数据可视化的功能。具有以下优点：

* 使用LightGBM算法快速准确地预测表型值，并支持GPU加速训练。
* 支持与表型相关的SNP的选择和可视化。
* 支持PCA和t-SNE两种降维算法来提取SNP信息。
* 支持Kmeans和OPTICS两种聚类算法来分析样本群体结构。
* 能够绘制基因型数据的杂合率、缺失率和等位基因频率的直方图。



## 文档

*英文文档*: [https://ibreeding.github.io](https://ibreeding.github.io)

*中文文档*: [https://ibreeding-ch.github.io](https://ibreeding-ch.github.io)



## 安装

### 通过conda安装（推荐）

```bash
$ conda install -c xu_cau_cab cropgbm 
```

### 通过pip安装

```bash
$ pip install --user cropgbm
```

### 通过源代码安装

```bash
$ tar -zxf CropGBM.tar.gz

# Install Python package dependencies of CropGBM: setuptools, wheel, numpy, scipy, pandas, scikit-learn, lightgbm, matplotlib, seaborn
$ pip install --user setuptools wheel numpy scipy pandas scikit-learn lightgbm matplotlib seaborn

# Install external dependencies of CropGBM: PLINK 1.90 
$ wget s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20191028.zip
$ mkdir plink_1.90
$ unzip plink_linux_x86_64_20191028.zip -d ./plink_1.90

# Add CropGBM, PLINK to the system environment variables for quick use:
$ vi ~/.bashrc
export PATH="/userpath/CropGBM:$PATH"
export PATH="/userpath/plink1.90:$PATH"
$ source ~/.bashrc
```


## 测试（通过conda）

进入 `/miniconda3/pkgs/cropgbm-1.1.2-py39_0/info/test` 文件夹

运行 `run_test.py` 程序来检查cropgbm是否能够在本地运行。



## 参数配置

CropGBM 支持 “配置文件” 与 “命令行” 两种参数赋值形式。CropGBM 会优先读取配置文件中各参数的值，再读取命令行中各参数的值。当某一参数被两种方式同时赋值时，CropGBM 以命令行中参数值为参考，忽略配置文件中的参数值。

```bash
# CropGBM 读取配置文件（-c config_path）中各参数的值并调用基因型数据预处理模块（-pg all）
$ cropgbm -c config_path -o ./gbm_result/ -pg all

# CropGBM 忽略配置文件中 fileformat 值，而以 ped 为参考
$ cropgbm -c config_path -o ./gbm_result/ -pg all --fileformat ped
```

**注意：**若程序无法运行，请尝试在程序名前添加 python。如 *$ python cropgbm -c config_path -pg all*

## 基因型数据预处理

基因型数据预处理模块的功能包括：提取指定 样本ID、snpID 的基因型数据，统计并直方图的形式展示 snp 缺失率、杂合率，基因型重编码等。为程序下游分析提供数据及可接受的文件格式。目前 CropGBM 支持的基因型文件输入格式有 ped、bed。

```bash
# 调用基因型数据预处理模块，统计并展示基因型数据的缺失率、杂合率等
$ cropgbm -o ./gbm_result/ -pg all --fileprefix genofile --fileformat ped
```



## 表型数据预处理

表型数据预处理模块的功能包括：提取指定 样本ID、snpID 的表型数据，表型归一化，表型重编码等。同时支持以直方图或箱线图的形式展示数据的分布情况。

```bash
# 调用表型数据预处理模块（-pp）进行归一化操作（--phe-norm）
$ cropgbm -o ./gbm_result/ -pp --phe-norm --phefile-path phefile.txt --phe-name DTT

# 根据样本所属的群体类别（--ppgroupfile-path groupfile.txt）提取表型数据并以箱线图的形式展示
$ cropgbm -o ./gbm_result/ -pp --phe-plot --phefile-path phefile.txt --phe-name DTT --ppgroupfile-path phefile.txt --ppgroupid-name paternal_line
```



## 群体结构分析

群体结构分析模块可以基于基因型数据分析样本的种群结构。CropGBM 支持使用 t-SNE 或 PCA 方法对基因型数据进行降维，使用 OPTICS 或 Kmeans 方法聚类。同时支持以散点图的形式展现样本的群体结构。

```bash
# 调用群体结构分析模块（-s），根据基因型数据对样本进行聚类并展示（--structure-plot）
$ cropgbm -o ./gbm_result/ -s --structure-plot --genofile-path genofile_filter.geno --redim-mode pca --cluster-mode kmeans --n-clusters 30
```



## 构建模型与特征选择

模型训练模块主要基于 lightGBM 算法编写而成。为提高模型的准确性，建议提供验证集辅助调参。若无验证集，可利用交叉验证来选择合适的参数值。CropGBM 根据训练模型中各 snp 的增益值筛选 snp。同时支持用箱线图和热图展示被筛选出的 snp 的重要性。

```bash
# 交叉验证（-e -cv）
$ cropgbm -o ./gbm_result/ -e -cv --traingeno train.geno --trainphe train.phe

# 构建训练模型（-e -t）。若无验证集数据，--validgeno 和 --validphe 参数可以省略。
$ cropgbm -o ./gbm_result/ -e -t --traingeno train.geno --trainphe train.phe --validgeno valid.geno --validphe valid.phe

# 特征选择（-e -t -sf），展示模型预测精度的变化（--bygain-boxplot）
$ cropgbm -o ./gbm_result/ -e -t -sf --bygain-boxplot --traingeno train.geno --trainphe train.phe
```



## 表型预测

表型预测模块利用模型训练模块输出的模型，预测测试集样本的表型。

```bash
# 表型预测（-e -p）
$ cropgbm -o ./gbm_result/ -e -p --testgeno test.geno --modelfile-path train.lgb_model
```

## 参考资料

https://ibreeding-ch.github.io/QuickStart
https://github.com/YuetongXU/CropGBM



