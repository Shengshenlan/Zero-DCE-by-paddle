# Zero-DCE-by-paddle

源项目地址：https://github.com/Li-Chongyi/Zero-DCE  
这里使用的paddle框架进行实现：  
https://aistudio.baidu.com/aistudio/projectdetail/5884162  
可以直接在上面的地址运行体验。

# 文件结构
Download the Zero-DCE_code first. The following shows the basic folder structure.

├── data  
│ ├── test_data # testing data. You can make a new folder for your testing data, like LIME, MEF, and NPE.  
│ │ ├── LIME  
│ │ └── MEF  
│ │ └── NPE  
│ └── train_data  
├── lowlight_test.py # testing code 
├── lowlight_train.py # training code  
├── model.py # Zero-DEC network  
├── dataloader.py  
├── snapshots  
│ ├── Epoch0.pdiparams # A pre-trained snapshot (Epoch0.pdiparams)  

# Test模式:
```
python lowlight_test.py
```
脚本将处理“test_data”文件夹子文件夹中的图像，并在“data”中创建一个新的文件夹“result”。您可以在“结果”文件夹中找到增强的图像。

# Train模式:
数据集可以从这里下载
https://aistudio.baidu.com/aistudio/datasetdetail/206295
```
python lowlight_train.py
```
