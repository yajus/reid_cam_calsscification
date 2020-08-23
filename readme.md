## 网络结构
backbone-Resnet50  
output-751  
no tirck  


### 特征提取网络
基于https://github.com/layumi/Person_reID_baseline_pytorch 修改  
网络代码：  
./v1/Person_reID_pytorch
### 分类数据集
prid_2011  
iLIDS-VID  
DukeMTMC-reID  
MSMT17  
market1501  
各数据集分类代码：  
./id  
### 基于特征的kmeans聚类
聚类代码：  
./kmeans.ipynb  
### Requirement
Python 3.7  
numpy  
pytorch  
scipy  
yaml  
panda  
