# ISRNet
本仓库是论文《轻量可逆的零样本图像超分网络》的项目代码。
ISRNet的超分结果以及对应的模型文件可自行下载。
+ [reuslts]()
+ [DIV2KRK数据集](https://github.com/sefibk/KernelGAN)

## 依赖包
```
Pytorch == 1.11.0
torchvision == 0.12.0
numpy == 1.21.5
scipy == 1.7.3
tqdm == 4.63.0
Pillow == 9.0.1
```

## 克隆项目
```
git clone https://github.com/Luision/ISRNet
```

## 测试
把测试的文件放置在test_images文件夹，在命令行输入以下代码，超分结果会输出在Results文件夹

```
# X2尺度
python train.py -i test_images/ -o Results/

# X4尺度
python train.py -i test_images/ -o Results/ --X4
```
