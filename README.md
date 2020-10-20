# IPN Hand: A Video Dataset and Benchmark for Real-Time Continuous Hand Gesture Recognition
## [Project page and download link of the dataset](https://gibranbenitez.github.io/IPN_Hand/)

PyTorch implementation, codes and pretrained models of the paper: 

[__IPN Hand: A Video Dataset and Benchmark for Real-Time Continuous Hand Gesture Recognition__](https://arxiv.org/abs/2005.02134)
<br>
[Gibran Benitez-Garcia](https://gibranbenitez.github.io), Jesus Olivares-Mercado, Gabriel Sanchez-Perez, and Keiji Yanai
<br>
___Accepted at [ICPR 2020](https://www.icpr2020.it/)___

This paper proposes the [__IPN Hand dataset__](https://gibranbenitez.github.io/IPN_Hand/), a new benchmark video dataset with sufficient size, variation, and real-world elements able to train and evaluate deep neural networks for continuous Hand Gesture Recognition (HGR).
With our dataset, the performance of three 3D-CNN models is evaluated on the tasks of isolated and continuous real-time HGR.
Since IPN hand contains RGB videos only, we analyze the possibility of increasing the recognition accuracy by adding multiple modalities derived from RGB frames, i.e., optical flow and semantic segmentation, while keeping the real-time performance. 

### Introduction video (supplementary material):

<div align="center" style="width:image width px;">
  <img src="https://img.youtube.com/vi/OH3n5rf2wV8/maxresdefault.jpg" href="https://youtu.be/OH3n5rf2wV8" width="640">
</div>

### Dataset details

The subjects from the dataset were asked to record gestures using their own PC keeping the defined resolution and frame rate. 
Thus, __only RGB videos__ were captured, and the distance between the camera and each subject varies.
All videos were recorded in the resolution of __640x480 at 30 fps__. 

Each subject continuously performed __21 gestures__ with three random breaks in a single video.
We defined [__13 gestures__](https://gibranbenitez.github.io/IPN_Hand/Classes) to control the pointer and actions focused on the interaction with touchless screens.

Description and statics of each gesture are shown in the next table. 
Duration is measured in the number of frames (30 frames = 1 s).

id |	Label |  Gesture	| Instances	| Mean duration (std)
-- | -------- | -------- | ---------- | -------------------
_1_ | _D0X_ | _Non-gesture_ | _1431_ |	_147 (133)_
2	| B0A | Pointing with one finger	| 1010	| 219 (67)
3  | B0B |	Pointing with two fingers	| 1007	| 224 (69)
4	| G01 | Click with one finger	| 200	| 56 (29)
5	| G02 | Click with two fingers | 200	| 60 (43)
6	| G03 | Throw up	| 200	| 62 (25)
7	| G04 | Throw down	| 201	| 65 (28)
8	| G05 | Throw left	| 200	| 66 (27)
9	| G06 | Throw right	| 200	| 64 (28)
10	| G07 | Open twice	| 200	| 76 (31)
11	| G08 | Double click with one finger	| 200	| 68 (28)
12	| G09 | Double click with two fingers	| 200	| 70 (30)
13	| G10 | Zoom in	| 200	| 65 (29)
14	| G11 | Zoom out	| 200	| 64 (28)
| |  | _All non-gestures:_	| _1431_	| _147 (133)_
| |   | _All gestures:_	| _4218_	| _140 (94)_
|  | |  ___Total:___	| ___5649___	| ___142 (105)___

#### [_Video examples of all classes (.GIF) here_](https://gibranbenitez.github.io/IPN_Hand/Classes)

### Baseline results

[___Baseline results___](https://gibranbenitez.github.io/IPN_Hand/Results) for isolated and continuous hand gesture recognition of the IPN Hand dataset can be found [__here__](https://gibranbenitez.github.io/IPN_Hand/Results).

## Requirements
Please install the following requirements.

- Python 3.5+
- PyTorch 1.0+
- TorchVision
- Pillow
- OpenCV

### Pretrained models
* [ResNeXt-101 models](https://drive.google.com/open?id=156fE3mO3YdFPY4pfreWYQn5sxQdu7Bmt) 
* [ResNet-50 models](https://drive.google.com/open?id=1X9uom_f0euHmhAgO8XNJUqUGH98saB7Z) 
* HarDNet model (soon)
* [Optical Flow model](https://github.com/sniklaus/pytorch-spynet) 


## Usage

### Preparation
* Download the dataset from [here](https://gibranbenitez.github.io/IPN_Hand/)
* Clone this repository
```console
$ git clone https://github.com/GibranBenitez/IPN-hand
```
* Store all pretrained models in `./report_ipn/`

### Isolated testing
* Change the path of the dataset from `./tests/run_offline_ipn_Clf.sh` and run
```bash
$ bash run_offline_ipn_Clf.sh
```
### Continuous testing
* Change the path of the dataset from `./tests/run_online_ipnTest.sh` and run
```bash
$ bash run_online_ipnTest.sh
```

## Citation
If you find useful the IPN Hand dataset for your research, please cite the paper:

```bibtex
@inproceedings{bega2020IPNhand,
  title={IPN Hand: A Video Dataset and Benchmark for Real-Time Continuous Hand Gesture Recognition},
  author={Benitez-Garcia, Gibran and Olivares-Mercado, Jesus and Sanchez-Perez, Gabriel and Yanai, Keiji},
  booktitle={25th International Conference on Pattern Recognition, {ICPR 2020}, Milan, Italy, Jan 10--15, 2021},
  pages={1--8},
  year={2021},
  organization={IEEE},
}
```
## Acknowledgement
This project is inspired by many previous works, including:
* [Real-time hand gesture detection and classification using convolutional neural networks](https://arxiv.org/abs/1901.10323), Kopuklu et al, _FG 2019_ [[code](https://github.com/ahmetgunduz/Real-time-GesRec)]
* [Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?](http://openaccess.thecvf.com/content_cvpr_2018/html/Hara_Can_Spatiotemporal_3D_CVPR_2018_paper.html), Hara et al, _CVPR 2018_ [[code](https://github.com/kenshohara/3D-ResNets-PyTorch)]
* [Optical Flow Estimation Using A Spatial Pyramid Network](https://arxiv.org/abs/1611.00850), Ranjan and Black, _CVPR 2017_ [[code](https://github.com/sniklaus/pytorch-spynet) by Niklaus]
* [HarDNet: A Low Memory Traffic Network](https://arxiv.org/abs/1909.00948), Chao et al, _ICCV 2019_ [[code](https://github.com/PingoLH/FCHarDNet)]
* [Learning to estimate 3d hand pose from single rgb images](http://openaccess.thecvf.com/content_iccv_2017/html/Zimmermann_Learning_to_Estimate_ICCV_2017_paper.html), Zimmermann and Brox, _ICCV 2017_ [[dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html)]
