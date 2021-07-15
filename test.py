import os
import json
import torch
import argparse
import numpy as np
from lib.network import get_autoencoder, fc
from lib.util.motion import preprocess_mixamo, preprocess_test, postprocess
from lib.util.general import get_config
from lib.data import get_dataloader
from lib.trainer import rotate_and_maybe_project_learning
from preprocess import testJoints2Video
from itertools import combinations
import matplotlib.pyplot as plt


def accuracy(y, x, true_label, pre_label):
    x = x.squeeze().cuda()
    y = y.squeeze().cuda()
    a = torch.sum(x ** 2, 1).unsqueeze(1)
    b = torch.sum(y ** 2, 1).unsqueeze(1).transpose(0, 1)
    c = 2 * torch.matmul(x, y.transpose(0, 1))
    dist = a + b - c
    dist = torch.sqrt(dist)
    idx = dist.sort(1)[1].cpu().numpy()
    #     print(x.shape,y.shape)
    print(idx.shape)
    pred_label = np.asarray([[true_label[idx[i][j]] for j in range(1) for i in range(len(idx))]]).squeeze()
    # np.save('pred_label.npy', pred_label)
    alist = []
    for i in range(len(pre_label)):
        print("true_label:{}\tpred_label:{}".format(pre_label[i], pred_label[i]))
        if pre_label[i] == pred_label[i]:
            alist.append(1)
    acc = sum(alist) / len(pred_label)
    # acc = sum(alist)
    return acc


def evaluate(autoencoder, fc, gallery_loader, probe_loader, feature_type=None):
    autoencoder.eval()
    fc.eval()

    flag = 0

    for gallery in gallery_loader:
        with torch.no_grad():  # 2D eval
            gallery_data = gallery['x'].float().cuda()
            gallery_label = gallery['label'].cuda()

            gallery_motion = autoencoder.encode_motion(gallery_data)
            gallery_body, gallery_body_seq = autoencoder.encode_body(gallery_data)
            gallery_view, gallery_view_seq = autoencoder.encode_view(gallery_data)

            if feature_type == "combined_all":
                gallery_features = torch.cat((gallery_data, gallery_motion, gallery_body_seq, gallery_view_seq), 1)
            if feature_type  == "combined":
                gallery_features = torch.cat((gallery_data, gallery_motion, gallery_body_seq), 1)
                # gallery_features = gallery_data
            elif feature_type == "motion":
                gallery_features = gallery_motion
            elif feature_type == "pose":
                gallery_features = gallery_data
            elif feature_type == "body":
                gallery_features = gallery_body_seq
            elif feature_type == "view":
                gallery_features = gallery_view_seq
            feature, out = fc(gallery_features, "gallery")

            if flag == 0:
                glabel = gallery_label
                gallery_feature = feature

                # max_motion, _ = torch.max(gallery_motion, 2)
                # gallery_feature = torch.cat((feature, max_motion, torch.squeeze(gallery_body, 2)), 1)
                flag = 1
            else:
                glabel = torch.cat((glabel, gallery_label), 0)

                # max_motion, _ = torch.max(gallery_motion, 2)
                # feature = torch.cat((feature, max_motion, torch.squeeze(gallery_body, 2)), 1)
                gallery_feature = torch.cat((gallery_feature, feature), 0)

    # gallery_all_feature = torch.cat((gallery_motion_feature, gallery_body_feature,
    #                                 gallery_pose_feature, gallery_view_feature), 1)

    flag = 0
    for probe in probe_loader:
        with torch.no_grad():  # 2D eval
            #
            probe_data = probe['x'].float().cuda()
            probe_label = probe['label'].cuda()
            #
            probe_motion = autoencoder.encode_motion(probe_data)
            probe_body, probe_body_seq = autoencoder.encode_body(probe_data)
            probe_view, probe_view_seq = autoencoder.encode_view(probe_data)

            if feature_type == "combined_all":
                probe_features = torch.cat((probe_data, probe_motion, probe_body_seq, probe_view_seq), 1)
            elif feature_type  == "combined":
                probe_features = torch.cat((probe_data, probe_motion, probe_body_seq), 1)
                # probe_features = probe_data
            elif feature_type == "motion":
                probe_features = probe_motion
            elif feature_type == "pose":
                probe_features = probe_data
            elif feature_type == "body":
                probe_features = probe_body_seq
            elif feature_type == "view":
                probe_features = probe_view_seq
            feature, out = fc(probe_features, "probe")

            if flag == 0:
                plabel = probe_label

                # max_motion, _ = torch.max(probe_motion, 2)
                # probe_feature = torch.cat((feature, max_motion, torch.squeeze(probe_body, 2)), 1)
                probe_feature = feature
                flag = 1
            else:
                plabel = torch.cat((plabel, probe_label), 0)

                # max_motion, _ = torch.max(probe_motion, 2)
                # feature = torch.cat((feature, max_motion, torch.squeeze(probe_body, 2)), 1)
                probe_feature = torch.cat((probe_feature, feature), 0)

    # probe_all_feature = torch.cat((probe_motion_feature, probe_body_feature,
    #                                probe_pose_feature, probe_view_feature), 1)

    acc = accuracy(gallery_feature, probe_feature, glabel, plabel)
    return acc


def test(gallery_mode = None, gallery_angle = None, probe_mode = None, probe_angle = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='which config to use.')
    parser.add_argument('--ae_checkpoint', type=str, required=True,
                        help="path to trained autoencoder model weights")
    parser.add_argument('--fc_checkpoint', type=str, required=True,
                        help="path to trained recognition model weights")
    args = parser.parse_args()

    config = get_config(args.config)
    ae = get_autoencoder(config)
    fcNet = fc(config.fc)
    ae.load_state_dict(torch.load(args.ae_checkpoint))
    fcNet.load_state_dict(torch.load(args.fc_checkpoint))
    ae.cuda()
    fcNet.cuda()
    print("loaded ae model")

    if gallery_mode != None:
        if gallery_angle != None:
            config.data.gallery_dir = config.data.gallery_dir + "gallery_" + gallery_mode + "_" + str(gallery_angle) + "_data.npy"
            config.data.gallery_label_dir = config.data.gallery_label_dir + "gallery_" + gallery_mode + "_" + str(
                gallery_angle) + "_label.npy"

    if probe_mode != None:
        if probe_angle != None:
            config.data.probe_dir = config.data.probe_dir + "probe_" + probe_mode + "_" + str(probe_angle) + "_data.npy"
            config.data.probe_label_dir = config.data.probe_label_dir + "probe_" + probe_mode + "_" + str(probe_angle) + "_label.npy"
        else:
            config.data.probe_dir = config.data.probe_dir + "probe_" + probe_mode + "_data.npy"
            config.data.probe_label_dir = config.data.probe_label_dir + "probe_" + probe_mode + "_label.npy"


    print(config.data.gallery_dir)
    print(config.data.probe_dir)
    probe_data = np.load(config.data.probe_dir)
    print(probe_data.shape)

    gallery_loader = get_dataloader("gallery", config)
    probe_loader = get_dataloader("probe", config)

    print(len(gallery_loader), len(probe_loader))
    all_acc = evaluate(ae, fcNet, gallery_loader, probe_loader, config.feature_type)
    print(all_acc)
    return all_acc

if __name__ == "__main__":

    gallery_mode = ["nm"]
    gallery_angle = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
    probe_mode = ["nm","bg","cl"]
    probe_angle = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
    test(gallery_mode[0], gallery_angle[0], probe_mode[0], probe_angle[0])

    result = np.zeros((3,11,11))
    for i in range(len(probe_mode)):
        mode = probe_mode[i]
        for j in range(len(probe_angle)):
            angle = probe_angle[j]
            for k in range(len(gallery_angle)):
                gl_angle = gallery_angle[k]
                acc = test(gallery_mode[0], gl_angle, mode, angle)
                result[i, k, j] = acc
    # 将result写入excel
    all_res = np.zeros((len(probe_mode), len(probe_angle)))
    for k in range(len(probe_mode)):
        statu = probe_mode[k]
        output = open("result/res_" + statu + "_result" +  ".xls", "w", encoding="gbk")
        output.write("gallery/probe\t000\t018\t036\t054\t072\t090\t108\t126\t144\t162\t180\tavg\n")
        for i in range(len(gallery_angle)):
            output.write(str(gallery_angle[i]))
            output.write("\t")
            for j in range(len(probe_angle)+1):
                if j != len(probe_angle):
                    output.write(str(result[k, i, j]))
                    output.write("\t")
                else:
                    tmp_res = (np.sum(result[k,i,:])-result[k,i,i])/10
                    all_res[k, i] = tmp_res
                    output.write(str(tmp_res))
            output.write("\n")
        output.close()
    print("nm:{}, bg:{}, cl:{}".format(np.average(all_res[0,:]), np.average(all_res[1,:]), np.average(all_res[2,:])))


