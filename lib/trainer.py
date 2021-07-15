import os
import torch
import torch.nn as nn
import numpy as np
import random
import lib.network
from lib.loss import *
from lib.util.general import weights_init, get_model_list, get_scheduler
from lib.network import fc
from preprocess import testJoints2Video

def rotate_and_maybe_project_learning(X, project_2d=False):
    """ 3d coordinate project to 2d """
    if project_2d:
        batch_size, channels, seq_len = X.size()
        n_joints = channels // 3
        X = X.reshape(batch_size, n_joints, 3, seq_len)
        X = X[:,:,[0,1],:]
        X = X.reshape(batch_size, n_joints*2, seq_len)
    return X

class BaseTrainer(nn.Module):
    def __init__(self, config):
        super(BaseTrainer, self).__init__()

        self.config = config

        autoencoder_cls = getattr(lib.network, config.autoencoder.cls)
        self.autoencoder = autoencoder_cls(config.autoencoder)
        self.discriminator = Discriminator(config.discriminator)
        self.fc = fc(config.fc)

        # Setup the optimizers
        beta1 = config.beta1
        beta2 = config.beta2
        dis_params = list(self.discriminator.parameters())
        ae_params = list(self.autoencoder.parameters())
        fc_params = list(self.fc.parameters())

        self.ae_opt = torch.optim.Adam([p for p in ae_params if p.requires_grad],
                                        lr=config.ae_lr, betas=(beta1, beta2), weight_decay=config.weight_decay)
        self.fc_opt = torch.optim.Adam([p for p in fc_params if p.requires_grad],
                                            lr=config.fc_lr, betas=(beta1, beta2), weight_decay=config.weight_decay)

        self.ae_scheduler = get_scheduler(self.ae_opt, config)
        self.fc_scheduler = get_scheduler(self.fc_opt, config)

        # Network weight initialization
        self.apply(weights_init(config.init))

    def forward(self, data):
        x_a, x_v = data["x_a"], data["x_v"]
        batch_size = x_a.size(0)
        self.eval()
        body_a, body_b = self.sample_body_code(batch_size)
        motion_a = self.autoencoder.encode_motion(x_a)
        body_a_enc, _ = self.autoencoder.encode_body(x_a)
        motion_b = self.autoencoder.encode_motion(x_v)
        body_b_enc, _ = self.autoencoder.encode_body(x_v)
        x_ab = self.autoencoder.decode(motion_a, body_b)
        x_ba = self.autoencoder.decode(motion_b, body_a)
        self.train()
        return x_ab, x_ba

    def dis_update(self, data, config):
        raise NotImplemented

    def ae_update(self, data, config):
        raise NotImplemented

    def fc_update(self, data, config):
        raise NotImplemented

    def recon_criterion(self, input, target):
        raise NotImplemented

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.ae_scheduler is not None:
            self.ae_scheduler.step()
        if self.fc_scheduler is not None:
            self.fc_scheduler.step()

    def resume(self, checkpoint_dir, config):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "autoencoder")
        state_dict = torch.load(last_model_name)
        self.autoencoder.load_state_dict(state_dict)
        iterations = int(last_model_name[-11:-3])
        # Load fc
        last_model_name = get_model_list(checkpoint_dir, "fc")
        state_dict = torch.load(last_model_name)
        self.fc.load_state_dict(state_dict)
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.ae_opt.load_state_dict(state_dict['autoencoder'])
        self.fc_opt.load_state_dict(state_dict['fc'])
        # Reinitilize schedulers
        self.ae_scheduler = get_scheduler(self.ae_opt, config, iterations)
        self.fc_scheduler = get_scheduler(self.fc_opt, config, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        ae_name = os.path.join(snapshot_dir, 'autoencoder_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'discriminator_%08d.pt' % (iterations + 1))
        fc_name = os.path.join(snapshot_dir, 'fc_%08d.pt' % (iterations + 1))
        torch.save(self.fc.state_dict(), fc_name)


        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        center_param_name = os.path.join(snapshot_dir, 'param.pt')
        torch.save(self.autoencoder.state_dict(), ae_name)
        torch.save(self.discriminator.state_dict(), dis_name)
        torch.save({'autoencoder': self.ae_opt.state_dict(), 'discriminator': self.dis_opt.state_dict(),
                    'fc': self.fc_opt.state_dict()}, opt_name)

    def validate(self, gallery_loader, probe_loader):
        acc = self.evaluate(gallery_loader, probe_loader)
        return acc

    def accuracy(self, y, x, true_label, pre_label):
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
        np.save('pred_label.npy', pred_label)
        alist = []
        for i in range(len(pre_label)):
            if pre_label[i] == pred_label[i]:
                alist.append(1)
        acc = sum(alist) / len(pred_label)
        # acc = sum(alist)
        return acc

    @staticmethod
    def recon_criterion(input, target):
        # print("input size: ", input.size())
        # print("target size: ", target.size())
        input = trans_to_cuda(input)
        target = trans_to_cuda(target)
        res = torch.mean(torch.pow((input - target),2))
        # print("recon_criterion: input:{}, target:{}, res:{}".format(input, target, res))
        return res

    def recon_criterion_align(self, input, target):
        cycle_len = 30 if input.shape[-1] > 30 else int(input.shape[-1]/2)
        B, keyp, T = input.size()
        loss = []
        for i in range(cycle_len):
            target_starti = 0
            target_endi = T - i -1
            input_starti = i
            input_endi = T - 1
            loss_i = self.recon_criterion(input[:,:,input_starti:input_endi], target[:,:,target_starti:target_endi])/(T-i)
            loss.append(loss_i)
        for i in range(cycle_len):
            target_starti = 0
            target_endi = T - i -1
            input_starti = i
            input_endi = T - 1
            loss_i = self.recon_criterion(target[:,:,input_starti:input_endi], input[:,:,target_starti:target_endi])/(T-i)
            loss.append(loss_i)
        return np.min(loss)

    def evaluate(self, gallery_loader, probe_loader):
        self.autoencoder.eval()
        self.fc.eval()

        flag = 0
        for gallery in gallery_loader:
            with torch.no_grad():  # 2D eval
                gallery_data = gallery['x'].float().cuda()
                gallery_label = gallery['label'].cuda()

                gallery_motion = self.autoencoder.encode_motion(gallery_data)
                gallery_body, gallery_body_seq = self.autoencoder.encode_body(gallery_data)
                gallery_view, gallery_view_seq = self.autoencoder.encode_view(gallery_data)

                if self.config.feature_type == "combined_all":
                    gallery_features = torch.cat((gallery_data, gallery_motion, gallery_body_seq, gallery_view_seq), 1)
                elif self.config.feature_type == "combined":
                    gallery_features = torch.cat((gallery_data, gallery_motion, gallery_body_seq), 1)
                elif self.config.feature_type == "motion":
                    gallery_features = gallery_motion
                elif self.config.feature_type == "body":
                    gallery_features = gallery_body_seq
                elif self.config.feature_type == "pose":
                    gallery_features = gallery_data
                elif self.config.feature_type == "view":
                    gallery_features = gallery_view_seq
                feature, out = self.fc(gallery_features, "gallery")

                if flag == 0:
                    glabel = gallery_label
                    gallery_feature = feature
                    flag = 1
                else:
                    glabel = torch.cat((glabel, gallery_label), 0)
                    gallery_feature = torch.cat((gallery_feature, feature), 0)


        flag = 0
        for probe in probe_loader:
            with torch.no_grad():  # 2D eval
        #
                probe_data = probe['x'].float().cuda()
                probe_label = probe['label'].cuda()
        #
                probe_motion = self.autoencoder.encode_motion(probe_data)
                probe_body, probe_body_seq = self.autoencoder.encode_body(probe_data)
                probe_view, probe_view_seq = self.autoencoder.encode_view(probe_data)

                if self.config.feature_type == "combined_all":
                    probe_features = torch.cat((probe_data, probe_motion, probe_body_seq, probe_view_seq), 1)
                elif self.config.feature_type == "combined":
                    probe_features = torch.cat((probe_data, probe_motion, probe_body_seq), 1)
                    # probe_features = probe_data
                elif self.config.feature_type ==  "motion":
                    probe_features = probe_motion
                elif self.config.feature_type == "body":
                    probe_features = probe_body_seq
                elif self.config.feature_type == "pose":
                    probe_features = probe_data
                elif self.config.feature_type == "view":
                    probe_features = probe_view_seq
                feature, out = self.fc(probe_features, "probe")

                if flag == 0:
                    plabel = probe_label
                    probe_feature = feature
                    flag = 1
                else:
                    plabel = torch.cat((plabel, probe_label), 0)
                    probe_feature = torch.cat((probe_feature, feature), 0)

        acc = self.accuracy(gallery_feature, probe_feature, glabel, plabel)
        return acc

class CasiabTrainer(BaseTrainer):
    def __init__(self, config):
        super(CasiabTrainer, self).__init__(config)

    def ae_update(self, data, iterations, config):

        x_a = data['x'].float().cuda()
        x_v = data['x_v'].float().cuda()
        self.ae_opt.zero_grad()

        # encode
        motion_a = self.autoencoder.encode_motion(x_a)
        body_a, body_a_seq = self.autoencoder.encode_body(x_a)
        view_a, view_a_seq = self.autoencoder.encode_view(x_a)
        motion_v = self.autoencoder.encode_motion(x_v)
        body_v, body_v_seq = self.autoencoder.encode_body(x_v)
        view_v, view_v_seq = self.autoencoder.encode_view(x_v)

        # time invariance loss
        self.loss_time_inv_body = config.body_time_weight * consecutive_cosine_similarity(body_a_seq)
        self.loss_time_inv_body_ls = config.body_time_ls_weight * consecutive_cosine_similarity(body_v_seq)

        # invariance loss(Done!)
        self.loss_inv_b_ls = config.inv_b_ls_w * (self.recon_criterion(body_a, body_v) if config.inv_b_ls_w > 0 else 0)
        self.loss_inv_m_ls = config.inv_m_ls_w * (self.recon_criterion(motion_a, motion_v) if config.inv_m_ls_w > 0 else 0)

        if config.is_stream:
            x_i = data["x_i"].float().cuda()
            motion_i = self.autoencoder.encode_motion(x_i)
            body_i, body_i_seq = self.autoencoder.encode_body(x_i)
            view_i, view_i_seq = self.autoencoder.encode_view(x_i)
            self.loss_inv_v_ls = config.inv_v_ls_w * (self.recon_criterion(view_a, view_i) if config.inv_v_ls_w > 0 else 0)
        else:
            self.loss_inv_v_ls = 0

        # reconstruction loss
        X_a_recon = self.autoencoder.decode(motion_a, body_a, view_a)
        x_a_recon = rotate_and_maybe_project_learning(X_a_recon, project_2d=True)
        X_v_recon = self.autoencoder.decode(motion_v, body_v, view_v)
        x_v_recon = rotate_and_maybe_project_learning(X_v_recon, project_2d=True)

        self.loss_recon = config.recon_x_w * (0.5 * self.recon_criterion_align(x_a_recon, x_a) + 0.5 * self.recon_criterion_align(x_v_recon, x_v))

        X_aw_recon = self.autoencoder.decode(motion_a, body_a, view_v)
        x_aw_recon = rotate_and_maybe_project_learning(X_aw_recon, project_2d=True)

        motion_aw = self.autoencoder.encode_motion(x_aw_recon)
        body_aw, body_aw_seq = self.autoencoder.encode_body(x_aw_recon)
        view_aw, view_aw_seq = self.autoencoder.encode_view(x_aw_recon)
        self.loss_inv_b_aw = config.inv_b_ls_w * (
            self.recon_criterion(body_a, body_aw) if config.inv_b_ls_w > 0 else 0)
        self.loss_inv_m_aw = config.inv_m_ls_w * (
            self.recon_criterion(motion_a, motion_aw) if config.inv_m_ls_w > 0 else 0)
        self.loss_inv_v_aw = config.inv_v_ls_w * (
            self.recon_criterion(view_v, view_aw) if config.inv_v_ls_w > 0 else 0)


        if config.is_stream:
            X_a_recon = self.autoencoder.decode(motion_a, body_a, view_i)
            x_a_recon = rotate_and_maybe_project_learning(X_a_recon, project_2d=True)
            X_i_recon = self.autoencoder.decode(motion_i, body_i, view_a)
            x_i_recon = rotate_and_maybe_project_learning(X_i_recon, project_2d=True)
            self.loss_recon_view = config.recon_x_w * (
                        0.5 * self.recon_criterion_align(x_a_recon, x_a) + 0.5 * self.recon_criterion_align(
                    x_i_recon, x_i))

        else:
            self.loss_recon_view = 0


        # motion特征拉伸 train4-18-triplet
        motion_a_rs, _ = torch.max(motion_a, 2)
        motion_v_rs, _ = torch.max(motion_v, 2)
        motion_i_rs, _ = torch.max(motion_i, 2)

        self.triplet_motion_loss = config.triplet_weight * TripletLoss(motion_a_rs, motion_v_rs, motion_i_rs)
        self.triplet_body_loss = config.triplet_weight * TripletLoss(body_a, body_v, body_i)
        self.triplet_view_loss =config.triplet_weight * TripletLoss(view_a, view_i, view_v)

        self.loss_total = torch.tensor(0.).float().cuda()
        self.loss_total += self.loss_recon
        self.loss_total += self.loss_recon_view
        self.loss_total += self.loss_inv_b_ls
        self.loss_total += self.loss_inv_m_ls
        self.loss_total += self.loss_time_inv_body + self.loss_time_inv_body_ls
        self.loss_total += self.loss_inv_v_ls
        self.loss_total += self.triplet_body_loss
        self.loss_total += self.triplet_motion_loss
        self.loss_total += self.triplet_view_loss
        self.loss_total.backward()
        self.ae_opt.step()
        return self.loss_total

    def get_features(self, x, config, phase):
        motion = self.autoencoder.encode_motion(x)
        body, body_seq = self.autoencoder.encode_body(x)
        view, view_seq = self.autoencoder.encode_view(x)
        # print("x shape:{}, motion shape:{}, body shape:{}, view shape:{}".format(x.shape, motion.shape,
        #                                                                          body_seq.shape, view_seq.shape))

        if config.feature_type == "combined_all":
            # print("x:{}, motion:{}, body:{}, view:{}".format(x, motion, body_seq, view_seq))
            features = torch.cat((x, motion, body_seq, view_seq), 1)
            # print("features shape: {}".format(features.shape))
        elif config.feature_type == "combined":
            features = torch.cat((x, motion, body_seq), 1)
        elif config.feature_type == "pose":
            features = x
        elif config.feature_type == "body":
            features = body_seq
        elif config.feature_type == "motion":
            features = motion
        elif config.feature_type == "view":
            features = view_seq

        feat, out = self.fc(features, phase)
        return feat, out

    def fc_update(self, data, config, phase):
        x = data['x'].float().cuda()
        label = data['label'].float().cuda()
        self.fc_opt.zero_grad()

        features, out = self.get_features(x, config, phase)

        loss = self.fc.cal_center_loss(features, label)
        self.center_loss = config.center_weight * loss
        self.id_loss = config.id_weight * self.fc.cal_id_loss(out, label)
        self.fc_loss = self.center_loss + self.id_loss

        print("center_loss: {}, id_loss: {}".format(self.center_loss, self.id_loss))

        self.fc_loss.backward()
        self.fc_opt.step()
        return out