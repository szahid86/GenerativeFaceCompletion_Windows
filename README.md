# GenerativeFaceCompletion_Windows
This repository presents the windows version of the Caffe for training and testing GenerativeFaceCompletion code (Yijunmaverick/GenerativeFaceCompletion)

[Yijunmaverick]() presented the MatCaffe implementation of their [paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Generative_Face_Completion_CVPR_2017_paper.pdf) at this [repo](https://github.com/Yijunmaverick/GenerativeFaceCompletion). Their code can directly be run on linux after installing the Caffe version. The official [Caffe](https://github.com/BVLC/caffe/tree/windows) version of windows is slightly different from the [Caffe](https://github.com/Yijunmaverick/GenerativeFaceCompletion/tree/master/include/caffe) version used by FaceCompletion, so the windows version of Caffe can not be used directly to do Face completion experiments on windows platform.

This repository contains the modified version of windows-Caffe. One can clone the repo on their windows system and start doing experiments with GenerativeFaceCompletion.

You need to follow some weird steps:

1- Download and install the dependencies mentioned on original [windows-Caffe](https://github.com/BVLC/caffe/tree/windows) repo. I am writing those steps here for convenience.

### Requirements

 - Visual Studio 2013 or 2015
     - Technically only the VS C/C++ compiler is required (cl.exe)
 - [CMake](https://cmake.org/) 3.4 or higher (Visual Studio and [Ninja](https://ninja-build.org/) generators are supported)

### Optional Dependencies

 - Python for the pycaffe interface. Anaconda Python 2.7 or 3.5 x64 (or Miniconda)
 - Matlab for the matcaffe interface.
 - CUDA 7.5 or 8.0 (use CUDA 8 if using Visual Studio 2015)
 - cuDNN v5

 We assume that `cmake.exe` and `python.exe` are on your `PATH`.
 
### Configuring Caffe

If git is installed on your windows system then you can use execute the following command to clone the original windows-Caffe repo:
```cmd
C:\> git clone https://github.com/BVLC/caffe.git
C:\> cd caffe
C:\caffe> git checkout windows
```

