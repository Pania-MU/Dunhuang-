import os
import torch
import random
import ast
import math
import cv2
import numpy as np
from PIL import Image, ImageFile
from collections import Counter
import torch.utils.data as data_utils
import torch.nn.functional as F

ImageFile.LOAD_TRUNCATED_IMAGES = True

def split_integer(m, n):
    assert n > 0
    quotient = int(m / n)
    remainder = m % n
    if remainder > 0:
        return [quotient] * (n - remainder) + [quotient + 1] * remainder
    if remainder < 0:
        return [quotient - 1] * -remainder + [quotient] * (n + remainder)
    return [quotient] * n

class Buddha_PathLoader(torch.utils.data.Dataset):
    def __init__(self, buddha_data_path):
        self.buddha_data_path = buddha_data_path

        buddha_image_list = []
        for i_root, i_dirs, i_files in os.walk(self.buddha_data_path + 'images/'):
            for i_file in i_files:
                buddha_image_list.append(i_file)

        buddha_label_list = []
        for l_root, l_dirs, l_files in os.walk(self.buddha_data_path + 'heads_labels/'):
            for l_file in l_files:
                buddha_label_list.append(l_file)

        buddha_image_list.sort()
        buddha_label_list.sort()

        self.buddha_id = []
        self.mbuddha_img_list = []
        self.mbuddha_head_img_list = []
        self.mbuddha_label_list = []
        self.mbuddha_head_label_list = []
        for m_label in buddha_label_list:
            m_h_label = m_label[:-6] + '_f.bmp'
            m_buddha = m_label[:-6] + '.jpg'
            self.buddha_id.append(m_label[:-6])
            self.mbuddha_img_list.append(buddha_data_path + 'images/' + m_buddha)
            self.mbuddha_label_list.append(buddha_data_path + 'figures_labels/' + m_h_label)
            self.mbuddha_head_label_list.append(buddha_data_path + 'heads_labels/' + m_label)

    def Load_buddha_list(self, folder_num=5, folder=0):
        s_num = len(self.mbuddha_img_list)
        mbuddha_head_dict_list = []
        folder_in_list = split_integer(s_num, folder_num)
        for idx in range(s_num):
            mb_id = self.buddha_id[idx]
            mb_img = self.mbuddha_img_list[idx]
            mb_label = self.mbuddha_label_list[idx]
            mb_h_label = self.mbuddha_head_label_list[idx]
            mbuddha_head_dict = {'img': mb_img, 'f_label': mb_label, 'h_label': mb_h_label, 'id': mb_id}
            mbuddha_head_dict_list.append(mbuddha_head_dict)

        tra_buddha_list = []
        val_buddha_list = []
        for folder_id in range(len(folder_in_list)):
            if folder_id == folder:
                val_folder_size = folder_in_list[folder_id]
                val_buddha_list = val_buddha_list + mbuddha_head_dict_list[:val_folder_size]
                mbuddha_head_dict_list = mbuddha_head_dict_list[val_folder_size:]
            else:
                tra_folder_size = folder_in_list[folder_id]
                tra_buddha_list = tra_buddha_list + mbuddha_head_dict_list[:tra_folder_size]
                mbuddha_head_dict_list = mbuddha_head_dict_list[tra_folder_size:]

        return tra_buddha_list, val_buddha_list

    def __getitem__(self, buddha_id):
        pass

    def __len__(self):
        pass

class BuddhaLoader(torch.utils.data.Dataset):
    def __init__(self, buddha_data_list, augmentation=False):
        self.buddha_data_list = buddha_data_list
        self.augmentation = augmentation

    def __getitem__(self, b_id):
        buddha_dct = self.buddha_data_list[b_id]
        buddha_img_path = buddha_dct['img']
        buddha_f_label_path = buddha_dct['f_label']
        buddha_h_label_path = buddha_dct['h_label']
        buddha_id = buddha_dct['id']
        # get buddha_img
        buddha_img = Image.open(buddha_img_path)
        buddha_img_array = np.array(buddha_img)
        buddha_img = torch.from_numpy(buddha_img_array)
        buddha_img = buddha_img.permute(2, 0, 1)
        # get buddha_figure label
        buddha_figure_label = Image.open(buddha_f_label_path)
        buddha_figure_label_array = np.array(buddha_figure_label)
        buddha_figure_label = torch.from_numpy(buddha_figure_label_array[:, :, :1])
        buddha_figure_label = buddha_figure_label.permute(2, 0, 1)
        # get buddha_head label
        buddha_head_label = Image.open(buddha_h_label_path)
        buddha_head_label_array = np.array(buddha_head_label)
        buddha_head_label = torch.from_numpy(buddha_head_label_array[:, :, :1])
        buddha_head_label = buddha_head_label.permute(2, 0, 1)

        if self.augmentation:
            _, height, width = buddha_head_label.size()
            as_h, as_w = height//2, width//3
            crop_y_sta = random.randrange(height-as_h)
            crop_y_end = crop_y_sta + as_h
            crop_x_sta = random.randrange(width-as_w)
            crop_x_end = crop_x_sta + as_w
            buddha_img = buddha_img[:, crop_y_sta:crop_y_end, crop_x_sta:crop_x_end]
            buddha_figure_label = buddha_figure_label[:, crop_y_sta:crop_y_end, crop_x_sta:crop_x_end]
            buddha_head_label = buddha_head_label[:, crop_y_sta:crop_y_end, crop_x_sta:crop_x_end]
        _, height, width = buddha_head_label.size()
        if height > width:
            scale = 512 / height
        else:
            scale = 512 / width
        # down sample
        buddha_img = F.interpolate(buddha_img.unsqueeze(0), scale_factor=scale, mode='nearest')[0]
        buddha_figure_label = (buddha_figure_label > 100).float()
        buddha_figure_label = F.interpolate(buddha_figure_label.unsqueeze(0),
                                            scale_factor=scale, mode='nearest')[0]
        buddha_head_label = (buddha_head_label > 100).float()
        buddha_head_label = F.interpolate(buddha_head_label.unsqueeze(0),
                                            scale_factor=scale, mode='nearest')[0]
        buddha_label = torch.clamp((buddha_figure_label + buddha_head_label * 100), 0, 2).long()[0]

        return buddha_img, buddha_label, buddha_id

    def __len__(self):
        return len(self.buddha_list)

class BuddhaSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(self.data_source)

    def __len__(self):
        return len(self.data_source)
