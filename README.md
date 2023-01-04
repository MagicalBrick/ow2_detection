# ow2_detection
 Experimental real-time tracking project using yolov5.

Recently launched yolov7, which is very interesting, it simplifies the code of yolov5 and
greatly improves the speed of inference. So I am wondering if I can use yolov5 to write a
real-time detection project, and use yolov7 to do real-time detection after successfully
learning the main principles of this project. This is the origin of this project~

最近推出了yolov7，这很有意思，它简化了yolov5的代码并且在很大程度上提高了推理的速度。所以我在想能不能用yolov5
写一个实时检测的项目，并且在成功学习到这个项目的主要原理之后运用yolov7做实时检测。这就是这个项目的由来啦~

## code reference 代码参考
My code mainly refers to the detection.py file in the yolov5 v7.0 official folder. Although
I still don't understand the algorithm principle of yolov5 very well, ordinary code
refactoring is enough.This code does not have the function of moving the mouse! Reject
cheating, start with you and me.

我的代码主要参考于yolov5 v7.0官方文件夹的detection.py文件，虽然还不是很理解yolov5的算法原理，不过普通的
代码重构是堪堪够用了。本代码没有挪动鼠标的功能！拒绝外挂，从你我做起。

## Instructions for use 使用说明
This warehouse contains an ow2 general enemy model. I chose the yolos model for its small
size and high frame rate. Running main.py can directly use the .pt model, but the detection
frame rate on the 2080 graphics card is about 30 frames. If you want to increase the frame
rate, please run the export.py file, You will get an engine model through pt model
conversion. Different graphics card computing power and various dependent package versions
will cause the engine model to be incompatible, so you must use the export.py file to
convert your own engine model, which will increase your detection frame rate to more than
60 frames.

这个仓库包含了一个ow2的敌方通用模型，为了体积小巧以及高帧率我选择了yolos模型。运行main.py可以直接使用.pt
模型，不过在2080显卡上的检测帧率约在30帧。如果想提高帧率，请运行export.py文件，你将会通过pt模型转换得到一
个engine模型。显卡算力以及各种依赖包版本不同会导致engine模型不通用，所以你必须得使用export.py文件转换出
你自己的engine模型。这将使你的检测帧率提高到60帧以上。

### NO accelerating inference
```python3 main.py --model ./ow2_s.pt```

### Accelerating inference with TRT

If you want to accelerate inference with TRT, you can run the following command line on anaconda.
This will create a virtual environment which allow you to use tensorTRT to export your xx.pt model to xx.engine model.

如果想运行export.py，可以在anaconda上运行以下命令行.
```
conda create -n yoloTRT python=3.8 
conda activate yoloTRT 
cd [path](example:D:\wo2_detection) 
pip install --upgrade setuptools pip wheel 
python -m pip install nvidia-pyindex
python -m pip install nvidia-cuda-runtime-cu11
conda install cuda -c nvidia/label/cuda-11.3.0 -c nvidia/label/cuda-11.3
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install tensorrt-8.4.0.6-cp38-none-win_amd64.whl
pip install onnx
pip install -r requirements.txt
conda deactivate
```
**Note: Please check the cuda version supported by your graphics card before using the above
code!**

**注意：在使用以上代码前请检测你的显卡支持的cuda版本！**

After the virtual environment is created and the dependencies are installed (all the above codes run without error), you can run export.py.

```
cd X:/.../ow2_detection
conda activate yoloTRT
python3 export.py
```

After this, we will get a .engine file. with this model, we can begin the detection now.And remember to switch back to the original environment. Because the virtual environment we created just now is just for TRT acceleration.

```conda deactivate```


**Please install the following dependencies in advance: cuda, cuDNN, tensorRT, and the
corresponding version of torch**

**请提前安装好以下依赖包：cuda, cuDNN, tensorRT，以及对应版本的torch**

Run the following code to install dependencies. If you encounter installation failure,
please run ```pip install **```. (*** is the dependent package name)

运行以下代码安装依赖包。如果遇到安装失败的情况，请运行```pip install **```。（***是依赖包名称）

```pip install -r requirements.txt```

After all dependent packages are installed, run the following code to run the program

```python3 main.py --model ow2_s.engine```



## demo video 演示视频
https://youtu.be/oBoWyB3-PEE

