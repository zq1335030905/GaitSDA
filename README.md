# Gait-recognition-with-disentanglement-features
This is the code for paper: Static and Dynamic Features Analysis from Human Skeletons for Gait Recognition in IJCB2021. If you have any qustions, you can contact with me by lzq.szu@gmail.com.


# Train
* bash
  ```sh
  CUDA_VISIBLE_DEVICES=2 python train.py --config configs/train.yaml --phase train
  ```

# Test
* bash 
  ```sh
  python test.py --config configs/test.yaml --ae_checkpoint out/checkpoints/autoencoder_00050000.pt --fc_checkpoint out/checkpoints/fc_00050000.pt
  ```

# Visualization
* bash 
  ```sh
  python visualize.py --config configs/test.yaml --checkpoint out/checkpoints/autoencoder_00050000.pt --heatmap 1 --exchange 1
  ```
"heatmap" and "exchange" can be set to 0 if you don't want to generate the results.
