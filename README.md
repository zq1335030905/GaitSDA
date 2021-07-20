# Gait-recognition-with-disentanglement-features
This is the code for paper: Static and Dynamic Features Analysis from Human Skeletons for Gait Recognition in IJCB2021. If you have any qustions, you can contact with me by lzq.szu@gmail.com.


# Train
'''sh
CUDA_VISIBLE_DEVICES=2 python train.py --config configs/train.yaml --phase train
'''
# Test
'''sh
CUDA_VISIBLE_DEVICES=0 python test.py --config configs/test.yaml --ae_checkpoint out/checkpoints/autoencoder_00050000.pt --fc_checkpoint out/checkpoints/fc_00050000.pt
'''

# Visualization


