import os
import json
from argparse import ArgumentParser

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

def get_parse():
    parser: ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='which config to use.')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="path to trained model weights")
    parser.add_argument('--save_folder', type=str, required=True, default="",
                        help="path to trained model weights")
    args = parser.parse_args()
    return args


def draw_heatmap(data, title_name):
    """
    :param data: [H, W]
    :return: None
    # 对H和W进行归一化到-1到1，然后用不同的颜色表
    """
    print(data)
    H, W = data.shape
    # for i in range(H):
    #     for j in range(W):
    #         data[i,j] = (data[i,j]-np.min(data[:,j]))/(np.max(data[:,j])-np.min(data[:,j]))
    x_label = [i for i in range(W)]
    y_label = [i for i in range(H)]

    print(data)
    fig = plt.figure(figsize=(12.8, 3))
    ax = fig.add_subplot(111)
    # ax.set_yticks(range(len(y_label)))
    # ax.set_yticklabels(y_label)
    # ax.set_xticks(range(len(x_label)))
    # ax.set_xticklabels(x_label)

    plt.xticks([])
    plt.yticks([])
    im = ax.imshow(data, cmap=plt.cm.viridis)
    plt.colorbar(im, orientation='horizontal')
    plt.title(title_name)
    plt.savefig(os.path.join("save_video/heatmap/", title_name + ".png"))
    plt.show()


def draw_heatmap_result(args):

    config = get_config(args.config)
    ae = get_autoencoder(config)
    ae.load_state_dict(torch.load(args.checkpoint))
    ae.cuda()
    ae.eval()
    print("Start draw heatmap result...")
    print("Loaded ae model")
    train_loader = get_dataloader("train", config)

    for it, data in enumerate(train_loader):
        x_a = data["x"]
        x_v = data["x_v"]
        x_i = data["x_i"]
        index = 0

        print("x_a label:{}, x_a_angle:{}".format(data["label"][index], data["x_angle"][index]))
        print("x_v label:{}, x_v_angle:{}, x_i_angle:{}".format(data["label"][index], data["x_v_angle"][index],
                                                                data["x_i_angle"][index]))
        motion_a = ae.encode_motion(x_a)
        body_a, body_a_seq = ae.encode_body(x_a)
        view_a, view_a_seq = ae.encode_view(x_a)

        motion_v = ae.encode_motion(x_v)
        body_v, body_v_seq = ae.encode_body(x_v)
        view_v, view_v_seq = ae.encode_view(x_v)

        motion_i = ae.encode_motion(x_i)
        body_i, body_i_seq = ae.encode_body(x_i)
        view_i, view_i_seq = ae.encode_view(x_i)

        # print(x_a[0,:,:].shape)
        # draw_heatmap(x_a[0,:,:].cpu().detach().numpy(), "pose of A")
        # draw_heatmap(x_v[0,:,:].cpu().detach().numpy(), "pose of V")
        # draw_heatmap(x_i[0,:,:].cpu().detach().numpy(), "pose of I")
        # print(body_a.shape)

        all_body = np.zeros((3, body_a[0].shape[0]))
        all_body[0,:] = torch.squeeze(body_a[0], 1).cpu().detach().numpy()
        all_body[1, :] = torch.squeeze(body_v[0], 1).cpu().detach().numpy()
        all_body[2, :] = torch.squeeze(body_i[0], 1).cpu().detach().numpy()

        all_view = np.zeros((3, view_a[0].shape[0]))
        all_view[0, :] = torch.squeeze(view_a[0], 1).cpu().detach().numpy()
        all_view[1, :] = torch.squeeze(view_v[0], 1).cpu().detach().numpy()
        all_view[2, :] = torch.squeeze(view_i[0], 1).cpu().detach().numpy()
        #
        draw_heatmap(all_body, "all_body")
        draw_heatmap(all_view, "all_view")

        #
        # draw_heatmap(motion_a[0].cpu().detach().numpy(), "motion of A")
        # draw_heatmap(motion_v[0].cpu().detach().numpy(), "motion of V")
        # draw_heatmap(motion_i[0].cpu().detach().numpy(), "motion of I")
        #
        draw_heatmap(np.abs((motion_a[0,:,:]-motion_v[0,:,:]).cpu().detach().numpy()), "motion diff of A and V")
        draw_heatmap(np.abs((motion_a[0,:,:]-motion_i[0,:,:]).cpu().detach().numpy()), "motion diff of A and I")
        break

    print("finished" + " " * 20)

def draw_exchange_result(args):

    config = get_config(args.config)
    ae = get_autoencoder(config)
    ae.load_state_dict(torch.load(args.checkpoint))
    ae.cuda()
    ae.eval()
    print("Loaded ae model")
    print("Start draw exchange result...")
    train_loader = get_dataloader("train", config)

    for it, data in enumerate(train_loader):
        x_a = data["x"]
        x_v = data["x_v"]
        index = 0
        save_folder = "test-18"

        print("x_a label:{}, x_a_angle:{}".format(data["label"][index], data["x_angle"][index]))
        print("x_v label:{}, x_v_angle:{}, x_i_angle:{}".format(data["label"][index], data["x_v_angle"][index],
                                                                data["x_i_angle"][index]))
        testJoints2Video(x_a[index], 0, False, save_folder)
        testJoints2Video(x_v[index], 1, False, save_folder)

        # print("x_a angle:{}, x_v angle:{}".format(data["x_angle"][index], data["x_v_angle"][index]))

        motion_a = ae.encode_motion(x_a)
        body_a, body_a_seq = ae.encode_body(x_a)
        view_a, view_a_seq = ae.encode_view(x_a)

        motion_v = ae.encode_motion(x_v)
        body_v, body_v_seq = ae.encode_body(x_v)
        view_v, view_v_seq = ae.encode_view(x_v)

        X_a_recon = ae.decode(motion_a, body_a, view_a)
        x_a_recon = rotate_and_maybe_project_learning(X_a_recon, project_2d=True)
        X_v_recon = ae.decode(motion_v, body_v, view_v)
        x_v_recon = rotate_and_maybe_project_learning(X_v_recon, project_2d=True)

        testJoints2Video(x_a_recon[index], 2, False, save_folder)
        testJoints2Video(x_v_recon[index], 3, False, save_folder)

        # across view
        X_a_recon = ae.decode(motion_a, body_a, view_v)
        x_a_recon = rotate_and_maybe_project_learning(X_a_recon, project_2d=True)
        X_v_recon = ae.decode(motion_v, body_v, view_a)
        x_v_recon = rotate_and_maybe_project_learning(X_v_recon, project_2d=True)

        testJoints2Video(x_a_recon[index], 4, False, save_folder)
        testJoints2Video(x_v_recon[index], 5, False, save_folder)

        break

    print("finished" + " " * 20)


if __name__ == '__main__':
    args = get_parse()
    if args.save_folder == "":
        draw_heatmap_result(args)
    else:
        draw_exchange_result(args)
        