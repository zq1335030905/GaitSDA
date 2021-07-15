from lib.data import get_dataloader
from lib.util.general import write_loss, get_config, to_gpu
import lib.trainer
import os
import time
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import tensorboardX
import shutil
import random
import string
from lib.data import CasiabDataset

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Path to the config file.")
    parser.add_argument("-o", "--out_dir", type=str, default="out", help="outputs path")
    parser.add_argument("-r", "--resume", action="store_true")
    parser.add_argument("-p", "--phase", type=str, default= "train", help="train or test")
    parser.add_argument("-a", "--ae_checkpoint", type=str, default=None, help="ae checkpoint path")
    parser.add_argument("-f", "--fc_checkpoint", type=str, default=None, help="fc checkpoint path")
    opts = parser.parse_args()
    return opts

def exclude_identity(acc):
    nm_acc = np.mean(np.sum(acc[0, :, :] - np.diag(np.diag(acc[0, :, :])), 1) / 10.0)
    bg_acc = np.mean(np.sum(acc[1, :, :] - np.diag(np.diag(acc[1, :, :])), 1) / 10.0)
    cl_acc = np.mean(np.sum(acc[2, :, :] - np.diag(np.diag(acc[2, :, :])), 1) / 10.0)
    avg_acc = (nm_acc + bg_acc + cl_acc) / 3.0
    return nm_acc, bg_acc, cl_acc, avg_acc

def train(config, opts, logger = None):
    for tmp in range(1):
        cudnn.benchmark = True

        # Load experiment setting
        max_iter = config.max_iter

        # Setup model and data loader
        train_data = CasiabDataset("train", config.data)
        trainer_cls = getattr(lib.trainer, config.trainer)
        trainer = trainer_cls(config)
        trainer.cuda()

        if opts.ae_checkpoint is not None and opts.resume is None:
            trainer.autoencoder.load_state_dict(torch.load(opts.ae_checkpoint))
            trainer.autoencoder.eval()
        if opts.fc_checkpoint is not None and opts.resume is None:
            trainer.fc.load_state_dict(torch.load(opts.fc_checkpoint))
            trainer.fc.eval()

        if logger is not None: logger.log("loading data")
        train_loader = get_dataloader("train", config)

        # Setup logger and output folders
        train_writer = tensorboardX.SummaryWriter(os.path.join(opts.out_dir, config.name, "logs", ''.join(random.sample(string.ascii_letters + string.digits, 8))))
        checkpoint_directory = os.path.join(opts.out_dir, config.name, 'checkpoints')
        os.makedirs(checkpoint_directory, exist_ok=True)
        shutil.copy(opts.config,
                    os.path.join(opts.out_dir, config.name, "config.yaml"))  # copy config file to output folder

        # Start training
        iterations = trainer.resume(checkpoint_directory, config=config) if opts.resume else 0

        pbar = tqdm(total=max_iter)
        pbar.set_description(config.name)
        pbar.update(iterations)
        print("%s: training started" % config.name)
        if logger is not None: logger.log("training started")

        raw_gallery_dir = config.data.gallery_dir
        raw_gallery_label_dir = config.data.gallery_label_dir
        raw_probe_dir = config.data.probe_dir
        raw_probe_label_dir = config.data.probe_label_dir

        start = time.time()
        while iterations < max_iter:
            train_acc = 0
            for it, data in enumerate(train_loader):
                phase = opts.phase
                if opts.ae_checkpoint is None:
                   trainer.ae_update(data, iterations, config)
                if opts.fc_checkpoint is None:
                    out = trainer.fc_update(data, config, phase)
                trainer.update_learning_rate()
                # train_correct = (out.cpu().detach().numpy() == data['label'].cpu().detach().numpy()).sum()

                # Run validation
                if (iterations + 1) % config.val_iter == 0:
                    # draw
                    # trainer.draw_result(data, config, iterations + 1)
                    gallery_angles = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
                    probe_angles = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
                    gallery_modes = ["nm"]
                    probe_modes = ["nm", "bg", "cl"]
                    all_acc = np.zeros((len(probe_modes), len(gallery_angles), len(probe_angles)))
                    for i in range(len(probe_modes)):
                        probe_mode = probe_modes[i]
                        for j in range(len(gallery_angles)):
                            gallery_angle = gallery_angles[j]
                            for k in range(len(probe_angles)):
                                probe_angle = probe_angles[k]
                                gallery_mode = gallery_modes[0]
                                if phase == "train":
                                    ext_gallery = "val_gallery_"
                                    ext_probe = "val_probe_"
                                elif phase == "test":
                                    ext_gallery = "gallery_"
                                    ext_probe = "probe_"
                                config.data.gallery_dir = raw_gallery_dir + ext_gallery + gallery_mode + "_" + str(
                                    gallery_angle) + "_data.npy"
                                config.data.gallery_label_dir = raw_gallery_label_dir + ext_gallery + gallery_mode + "_" + str(
                                    gallery_angle) + "_label.npy"
                                config.data.probe_dir = raw_probe_dir + ext_probe + probe_mode + "_" + str(
                                    probe_angle) + "_data.npy"
                                config.data.probe_label_dir = raw_probe_label_dir + ext_probe + probe_mode + "_" + str(
                                    probe_angle) + "_label.npy"
                                gallery_loader = get_dataloader("gallery", config)
                                probe_loader = get_dataloader("probe", config)
                                print("Val@ probe_mode:{}, gallery_angle:{}, probe_angle:{}".format(probe_mode, gallery_angle, probe_angle))
                                all_acc[i, j, k] = trainer.validate(gallery_loader, probe_loader)
                                print("acc:{}".format(all_acc[i,j,k]))
                    avg_acc = np.mean(all_acc)
                    train_writer.add_scalar("avg_acc(include)", avg_acc, iterations + 1)
                    nm_acc, bg_acc, cl_acc, avg_acc = exclude_identity(all_acc)
                    print('===Rank-%d (Exclude identical-view cases)===' % (1))
                    print('NM: {}, BG: {}, CL: {} '.format(nm_acc, bg_acc, cl_acc))
                    train_writer.add_scalar("avg_acc(exclude)", avg_acc, iterations + 1)

                # Dump training stats in log file
                if (iterations + 1) % config.log_iter == 0:
                    if logger is not None:
                        elapsed = (time.time() - start) / 3600.0
                        logger.log("training %6d/%6d, elapsed: %.2f hrs" % (iterations + 1, max_iter, elapsed))
                    write_loss(iterations, trainer, train_writer)

                # Save network weights
                if (iterations + 1) % config.snapshot_save_iter == 0:
                    trainer.save(checkpoint_directory, iterations)

                iterations += 1
                pbar.update(1)

            # train_acc += train_correct.data[0]
            # train_acc /= len(train_data)
            # train_writer.add_scalar("train_acc", train_acc, iterations + 1)

if __name__ == "__main__":
    opts = parse_args()
    config = get_config(opts.config)
    train(config, opts)
