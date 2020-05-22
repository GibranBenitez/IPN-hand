# IPN Hand: A Video Dataset and Benchmark for Real-Time Continuous Hand Gesture Recognition
PyTorch implementation, codes and pretrained models of the paper: 

[__IPN Hand: A Video Dataset and Benchmark for Real-Time Continuous Hand Gesture Recognition__](https://arxiv.org/abs/2005.02134)
<br>
[Gibran Benitez-Garcia](https://gibranbenitez.github.io), Jesus Olivares-Mercado, Gabriel Sanchez-Perez, and Keiji Yanai
<br>
arXiv 2020

This paper proposes the [__IPN Hand dataset__](https://gibranbenitez.github.io/IPN_Hand/), a new benchmark video dataset with sufficient size, variation, and real-world elements able to train and evaluate deep neural networks for continuous Hand Gesture Recognition (HGR).
With our dataset, the performance of three 3D-CNN models is evaluated on the tasks of isolated and continuous real-time HGR.
Since IPN hand contains RGB videos only, we analyze the possibility of increasing the recognition accuracy by adding multiple modalities derived from RGB frames, i.e., optical flow and semantic segmentation, while keeping the real-time performance. 

### Introduction video (supplementary material):

<div align="center" style="width:image width px;">
  <img src="https://img.youtube.com/vi/OH3n5rf2wV8/maxresdefault.jpg" href="https://youtu.be/OH3n5rf2wV8" width="640">
</div>

## [Project page and download link of the dataset](https://gibranbenitez.github.io/IPN_Hand/)

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
@article{bega2020IPNhand,
  title={IPN Hand: A Video Dataset and Benchmark for Real-Time Continuous Hand Gesture Recognition},
  author={Benitez-Garcia, Gibran and Olivares-Mercado, Jesus and Sanchez-Perez, Gabriel and Yanai, Keiji},
  journal={arXiv preprint arXiv:2005.02134},
  year={2020}
}
```
## Acknowledgement
This project is inspired by many previous works, including:
* [Real-time hand gesture detection and classification using convolutional neural networks](https://arxiv.org/abs/1901.10323), Kopuklu et al, _FG 2019_ [[code](https://github.com/ahmetgunduz/Real-time-GesRec)]
* [Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?](http://openaccess.thecvf.com/content_cvpr_2018/html/Hara_Can_Spatiotemporal_3D_CVPR_2018_paper.html), Hara et al, _CVPR 2018_ [[code](https://github.com/kenshohara/3D-ResNets-PyTorch)]
* [Optical Flow Estimation Using A Spatial Pyramid Network](https://arxiv.org/abs/1611.00850), Ranjan and Black, _CVPR 2017_ [[code](https://github.com/sniklaus/pytorch-spynet) by Niklaus]
* [HarDNet: A Low Memory Traffic Network](https://arxiv.org/abs/1909.00948), Chao et al, _ICCV 2019_ [[code](https://github.com/PingoLH/FCHarDNet)]
* [Learning to estimate 3d hand pose from single rgb images](http://openaccess.thecvf.com/content_iccv_2017/html/Zimmermann_Learning_to_Estimate_ICCV_2017_paper.html), Zimmermann and Brox, _ICCV 2017_ [[dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html)]
