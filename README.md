# IPN Hand: A Video Dataset and Benchmark for Real-Time Continuous Hand Gesture Recognition

PyTorch implementation of our arXiv paper [IPN Hand: A Video Dataset and Benchmark for Real-Time Continuous Hand Gesture Recognition](https://arxiv.org/abs/2005.02134), codes and pretrained models.

### Introduction video (supplementary material):

<div align="center" style="width:image width px;">
  <img src="https://img.youtube.com/vi/OH3n5rf2wV8/maxresdefault.jpg" href="https://youtu.be/OH3n5rf2wV8" width="640">
</div>
## [Project page and download link of the dataset](https://gibranbenitez.github.io/IPN_Hand/)

## Abstract
Continuous hand gesture recognition (HGR) is an essential part of human-computer interaction with a wide range of applications in the automotive sector, consumer electronics, home automation, and others. In recent years, accurate and efficient deep learning models have been proposed for HGR. However, in the research community, the current publicly available datasets lack real-world elements needed to build responsive and efficient HGR systems. In this paper, we introduce a new benchmark dataset named IPN Hand with sufficient size, variation, and real-world elements able to train and evaluate deep neural networks. This dataset contains more than 4 000 gesture samples and 800 000 RGB frames from 50 distinct subjects. We design 13 different static and dynamic gestures focused on interaction with touchless screens. We especially consider the scenario when continuous gestures are performed without transition states, and when subjects perform natural movements with their hands as non-gesture actions. Gestures were collected from about 30 diverse scenes, with real-world variation in background and illumination. With our dataset, the performance of three 3D-CNN models is evaluated on the tasks of isolated and continuous real-time HGR. Furthermore, we analyze the possibility of increasing the recognition accuracy by adding multiple modalities derived from RGB frames, i.e., optical flow and semantic segmentation, while keeping the real-time performance of the 3D-CNN model. Our empirical study also provides a comparison with the publicly available nvGesture (NVIDIA) dataset. The experimental results show that the state-of-the-art ResNext-101 model decreases about 30% accuracy when using our real-world dataset, demonstrating that the IPN Hand dataset can be used as a benchmark, and may help the community to step forward in the continuous HGR.

### Baseline models will be available soon
