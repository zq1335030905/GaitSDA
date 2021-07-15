import sys, os
thismodule = sys.modules[__name__]
import torch
import glob
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from preprocess import testJoints2Video

view_angles = np.array([ i * np.pi / 6 for i in range(-3, 4)])

def get_dataloader(phase, config):

    config.data.batch_size = config.batch_size
    config.data.seq_len = config.seq_len
    dataset_cls_name = config.data.train_cls if phase == 'train' else config.data.eval_cls
    dataset_cls = getattr(thismodule, dataset_cls_name)
    dataset = dataset_cls(phase, config.data)
    print(len(dataset))
    dataloader = DataLoader(dataset, shuffle=(phase=='train'),
                            batch_size=config.batch_size,
                            # num_workers=(config.data.num_workers if phase == 'train' else 1),
                            num_workers=0,
                            worker_init_fn=lambda _: np.random.seed(),
                            drop_last=False)
    return dataloader

class CasiabDataset(Dataset):

    def __init__(self, phase, config):
        super(CasiabDataset, self).__init__()

        assert phase in ['train', 'gallery', 'probe', "train_gallery", "train_probe"]
        self.config = config
        self.phase = phase
        self.seq_len = config.seq_len

        if phase == "train":
            self.data_path = config.train_dir
            self.label_path = config.train_label_dir
            self.angle_path = config.train_angle_dir
            self.situation_path = config.train_situation_dir
        elif phase == "gallery":
            self.data_path = config.gallery_dir
            self.label_path = config.gallery_label_dir
        elif phase == "probe":
            self.data_path = config.probe_dir
            self.label_path = config.probe_label_dir
        else:
            raise NotImplementedError

        # data include [B, T, 15, 2]
        self.data = np.load(self.data_path)
        self.label = np.load(self.label_path)-1
        if phase == "train":
            self.angle = np.load(self.angle_path)
            self.situation = np.load(self.situation_path)
            # 增加6倍训练数据量，iteration也要相应增加
            diff_angles = [18,36,54,72,90,18,36,54,72,90,18,36,54,72,90]
            self.view_data, self.view_data_angle = self.GetmatchData(self.data, self.angle, self.situation, self.label,
                                                                     diff_angle = diff_angles[0])

            if len(diff_angles) >1:
                for diff_angle in diff_angles[1:]:
                    view_data, view_data_angle = self.GetmatchData(self.data, self.angle, self.situation, self.label,
                                                                         diff_angle)
                    self.view_data = np.concatenate((self.view_data, view_data), 0)
                    self.view_data_angle = np.concatenate((self.view_data_angle, view_data_angle), 0)
                data = self.data
                angle = self.angle
                label = self.label
                for i in range(len(diff_angles)-1):
                    self.data = np.concatenate((self.data, data), 0)
                    self.label = np.concatenate((self.label, label), 0)
                    self.angle = np.concatenate((self.angle, angle), 0)

            if self.config.stream == 3:
                self.id_data, self.id_data_angle = self.GetmatchData2()

    def dimchange_fix(self, data):
        try:
            T, keyp, dim = data.shape
        except:
            return data
        data = torch.from_numpy(data)
        data = data.view(keyp * dim, T)
        data = np.array(data)
        return data

    def preprocess_fix(self, data):
        T, dim, keypoint = data.shape
        true_t = T
        fdata = data
        for t in range(T):
            if data[t,7,0] == 0.0 and data[t,6,0] == 0.0 and data[t,13,0] == 0.0 and data[t,14,0] == 0.0:
                true_t = t
                if true_t == 0:
                    return []
                break
        if true_t<self.seq_len/2:
            fdata[true_t:2*true_t,:,:] = data[:true_t,:,:]
            fdata[2*true_t:self.seq_len,:,:] = data[:self.seq_len-2*true_t,:,:]
            fdata = fdata[:self.seq_len, :, :]
        elif true_t<self.seq_len:
            fdata[true_t:self.seq_len,:,:] = data[:self.seq_len-true_t,:,:]
            fdata = fdata[:self.seq_len,:,:]
        elif true_t>self.seq_len:
            # 生成随机的序列
            random_idxes = sorted(random.sample(range(0, true_t), self.seq_len))
            random_list = np.zeros((self.seq_len, dim, keypoint))
            for i in range(len(random_idxes)):
                random_list[i,:,:] = fdata[random_idxes[i],:,:]
            fdata = random_list
        else:
            fdata = fdata[:self.seq_len,:,:]
        return fdata

    def GetmatchData(self, data, angle, situation, label, diff_angle):
        """
        get same id and different view data
        """
        view_data = np.zeros(shape=data.shape)
        # print("Generating view data!")
        view = np.zeros(shape=data.shape[0])
        for index in range(data.shape[0]):
            for randindex in range(-22, 22):
                try:
                    if abs(angle[index-randindex] - angle[index]) == diff_angle and label[index-randindex] == label[index]:
                        view_data[index] = data[index-randindex]
                        view[index] = int(angle[index-randindex])
                except:
                    if abs(angle[index+randindex] - self.angle[index]) == diff_angle and label[index+randindex] == label[index]:
                        view_data[index] = data[index+randindex]
                        view[index] = int(angle[index+randindex])
        # print("Finish generating view data!")
        return view_data, view

    def GetmatchData2(self):
        """
        get different id but same view data
        """
        view_data = np.zeros(shape=self.data.shape)
        # print("Generating view data!")
        view = np.zeros(shape=self.data.shape[0])
        # 随机寻找id
        random_idx = random.randint(121, self.data.shape[0]-121)
        for index in range(self.data.shape[0]):
            for randindex in range(random_idx, random_idx + 120):
                try:
                    if abs(self.angle[index-randindex] - self.angle[index]) == 0 and self.label[index-randindex] != self.label[index]:
                        view_data[index] = self.data[index-randindex]
                        view[index] = int(self.angle[index-randindex])
                except:
                    if abs(self.angle[index+randindex] - self.angle[index]) == 0 and self.label[index+randindex] != self.label[index]:
                        view_data[index] = self.data[index+randindex]
                        view[index] = int(self.angle[index+randindex])
        # print("Finish generating view data!")
        return view_data, view

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x = self.data[index]
        x = self.preprocess_fix(x)
        x = self.dimchange_fix(x)

        label = self.label[index]
        if self.phase == "train":

            x_angle = self.angle[index]
            x_v = self.view_data[index]
            x_v_angle = self.view_data_angle[index]

            x_v = self.preprocess_fix(x_v)
            x_v = self.dimchange_fix(x_v)

            x_i = self.id_data[index]
            x_i = self.preprocess_fix(x_i)
            x_i = self.dimchange_fix(x_i)
            x_i_angle = self.id_data_angle[index]

            if x == [] or x_v == [] or x_i == []:
                return self.__getitem__(index + 1)
            elif True in np.isnan(x) or True in np.isnan(x_v) or True in np.isnan(x_i):
                return self.__getitem__(index + 1)
            else:
                return {"x": x, "x_v": x_v, "x_i": x_i, "x_angle": x_angle, "x_v_angle": x_v_angle,
                        "x_i_angle": x_i_angle, "label": label}

        else:
            return {"x": x, "label": label}



