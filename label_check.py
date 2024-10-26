import os
import torch
import numpy as np
import shutil
from PIL import Image, ImageFile

buddha_data_path = '/DISK0/DATA base/Buddhas/OneDrive-2024-08-29/'

buddha_image_list = []
for i_root, i_dirs, i_files in os.walk(buddha_data_path + 'images/'):
    for i_file in i_files:
        buddha_image_list.append(i_file)

buddha_label_list = []
for l_root, l_dirs, l_files in os.walk(buddha_data_path + 'labels/'):
    for l_file in l_files:
        buddha_label_list.append(l_file)

for m_buddha, m_label in zip(buddha_image_list, buddha_label_list):
    label_list = open(buddha_data_path + 'labels/' + m_label, 'r')
    label_in_this_image = label_list.readlines()

    buddha_image_list.sort()
    buddha_label_list.sort()

for m_buddha in buddha_image_list:
    m_label = m_buddha[:-4] + '.txt'
    # get buddha figures and heads label lists
    label_path = buddha_data_path + 'labels/' + m_label
    if not os.path.exists(label_path):
        print(m_label)