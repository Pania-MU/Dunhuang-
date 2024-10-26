import os
import torch
import numpy as np
import shutil
from PIL import Image, ImageFile

buddha_data_path = '/DISK0/DATA base/Buddhas/OneDrive-2024-08-29/'
figure_size_rate = 0.7

# #
# buddha_image_path = buddha_data_path + 'images/'
# # 获取父文件夹中的所有子文件夹
# sub_dirs = [os.path.join(buddha_image_path, d) for d in os.listdir(buddha_image_path) if os.path.isdir(os.path.join(buddha_image_path, d))]
# # 遍历每个子文件夹
# for sub_dir in sub_dirs:
#     # 获取子文件夹中的文件列表
#     files = [os.path.join(sub_dir, f) for f in os.listdir(sub_dir) if os.path.isfile(os.path.join(sub_dir, f))]
#     # 遍历每个文件并移动到上一级文件夹
#     for file in files:
#         shutil.move(file, buddha_image_path)
#
# buddha_label_path = buddha_data_path + 'labels/'
# # 获取父文件夹中的所有子文件夹
# sub_dirs = [os.path.join(buddha_label_path, d) for d in os.listdir(buddha_label_path) if os.path.isdir(os.path.join(buddha_label_path, d))]
# # 遍历每个子文件夹
# for sub_dir in sub_dirs:
#     # 获取子文件夹中的文件列表
#     files = [os.path.join(sub_dir, f) for f in os.listdir(sub_dir) if os.path.isfile(os.path.join(sub_dir, f))]
#     # 遍历每个文件并移动到上一级文件夹
#     for file in files:
#         shutil.move(file, buddha_label_path)

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

for m_label in buddha_label_list:
    m_buddha = m_label[:-4] + '.jpg'
    # get m buddha image
    img = Image.open(buddha_data_path + 'images/' + m_buddha)
    mbuddha_labels = np.zeros((img.size[1], img.size[0]))
    mbuddha_head_labels = np.zeros((img.size[1], img.size[0]))
    # get buddha figures and heads label lists
    label_list = open(buddha_data_path + 'labels/' + m_label, 'r')
    label_list_in_image = label_list.readlines()
    # one hot labels
    for labels_in_image in label_list_in_image:
        # translate str list to float list
        labels_in_str = labels_in_image.split(' ')
        labels_in_float = []
        for p_str in labels_in_str:
            # print(p_str)
            labels_in_float.append(float(p_str))
        # get size param
        i_width, i_height = img.size
        t_height = int(i_height * labels_in_float[4])
        t_width = int(i_width * labels_in_float[3])
        start_y = int(i_height * labels_in_float[2] - (t_height // 2))
        start_x = int(i_width * labels_in_float[1] - (t_width // 2))
        if labels_in_image[0] == '0':
            new_height = int(figure_size_rate * t_height)
            y_edge = int(t_height - new_height)//2
            new_width = int(figure_size_rate * t_width)
            x_edge = int(t_width - new_width)//2
            mbuddha_labels[start_y+y_edge:start_y+y_edge+new_height, start_x+x_edge:start_x+x_edge+new_width] = 255
        else:
            mbuddha_head_labels[start_y:start_y+t_height, start_x:start_x+t_width] = 255
    # save one hot labels
    pil_mbuddha_labels = Image.fromarray(mbuddha_labels).convert("RGBA")
    pil_mbuddha_head_labels = Image.fromarray(mbuddha_head_labels).convert("RGBA")
    figures_save_path = buddha_data_path + 'figures_labels/'
    heads_save_path = buddha_data_path + 'heads_labels/'
    pil_mbuddha_labels.save(figures_save_path + m_buddha[:-4] + '_f.bmp')
    pil_mbuddha_head_labels.save(heads_save_path + m_label[:-4] + '_h.bmp')
    print(m_buddha[:-4] + ' | ' + m_label[:-4])

    # # check label
    # buddha_head_img_array = np.array(img)
    # buddha_head = torch.from_numpy(buddha_head_img_array)
    #
    # out_tensor = buddha_head * 0.5
    # mbuddha_labels_tensor = torch.from_numpy(mbuddha_labels).unsqueeze(-1).long()
    # mbuddha_head_labels_tensor = torch.from_numpy(mbuddha_head_labels).unsqueeze(-1).long()
    # ml_pp = torch.count_nonzero(mbuddha_labels_tensor)
    # mhl_pp = torch.count_nonzero(mbuddha_head_labels_tensor)
    # ml_max = mbuddha_labels_tensor.max()
    # mhl_max = mbuddha_head_labels_tensor.max()
    # out_tensor[:, :, 0:1] = out_tensor[:, :, 0:1] + mbuddha_labels_tensor
    # out_tensor[:, :, 1:2] = out_tensor[:, :, 1:2] + mbuddha_head_labels_tensor
    # out_tensor = out_tensor.clamp(min=0, max=255)
    # out_np = out_tensor.numpy().astype(np.uint8)
    # out_pil = Image.fromarray(out_np)
    # check_save_path = buddha_data_path + 'check/'
    # out_pil.save(check_save_path + m_buddha[:-4] + '_c.bmp')