## Gait recognition Static and Dynamic Features Analysis(Pytorch)
This is the code for paper: Static and Dynamic Features Analysis from Human Skeletons for Gait Recognition in 2021 IEEE International Joint Conference on Biometrics(IJCB). If you have any qustions, you can contact with me by lzq.szu@gmail.com. The introduction video is released in [bilibili](https://www.bilibili.com/video/BV1uq4y1p7jc/) and [youtube](https://www.youtube.com/watch?v=D4WiPNLJWcc&t=1s&ab_channel=%E6%9D%8E%E5%AD%90%E7%90%BC). Paper link [here](https://ieeexplore.ieee.org/abstract/document/9484378).

## About The Project
<img width="960" height="250" src="https://github.com/zq1335030905/Gait-recognition-with-disentanglement-features/blob/main/imgs/model_architecture_fixed.jpg"/>
Gait recognition is an effective way to identify a person due to its non-contact and long-distance acquisition. In addition, the length of human limbs and the motion pattern
of human from human skeletons have been proved to be effective features for gait recognition. However, the length of human limbs and motion pattern are calculated through
human prior knowledge, more important or detailed information may be missing. Our method proposes to obtain the dynamic information and static information from human skeletons through disentanglement learning. In the experiments, it has been shown that the features extracted by our method are effective.

## Prepare for data
You should download the data [CASIA-B](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp), and extracted by alphapose or openpose, our network input x ([Batch, 30, 64]). 30 = 15*2, 15 is the num of keypoints and 2 is the x and y coordinate of each keypoint.

## Getting Start
### Environment
```sh
python 3.6.9
pytorch 
tensorboard
```
### Train
  We train the disentanglement module and recognition module at the same time.
  ```sh
  CUDA_VISIBLE_DEVICES=2 python train.py --config configs/train.yaml --phase train
  ```
  
### Tensorboard
  You can visualize the training process in tensorboard, and you should change the PortId, for example, 8008. 
  ```sh
  tensorboard --logdir out/logs/ --port PortId
  ```

### Test
The pretrained model parameters can be download in [BaiduYunPan](https://pan.baidu.com/s/1CabNogyq_DoO8W2CWxfMSQ), the extract code is 8652. You should download the parameters of fc and autoencoder to directory "out/checkpoints/".
  ```sh
  python test.py --config configs/test.yaml --ae_checkpoint out/checkpoints/autoencoder_00050000.pt --fc_checkpoint out/checkpoints/fc_00050000.pt
  ```
The test result will be saved in an excel file.

### Visualization
  ```sh
  python visualize.py --config configs/test.yaml --checkpoint out/checkpoints/autoencoder_00050000.pt --heatmap 1 --exchange 1
  ```
"heatmap" and "exchange" can be set to 0 if you don't want to generate the results.

### heatmap
motion difference.

<img width="500" height="490" src="https://github.com/zq1335030905/Gait-recognition-with-disentanglement-features/blob/main/imgs/motion.jpg"/>

body and view features.

<img width="600" height="150" src="https://github.com/zq1335030905/Gait-recognition-with-disentanglement-features/blob/main/imgs/bodyandview.jpg"/>

view exchange.

<img width="600" height="300" src="https://github.com/zq1335030905/Gait-recognition-with-disentanglement-features/blob/main/imgs/view-disentanglement.jpg"/>
