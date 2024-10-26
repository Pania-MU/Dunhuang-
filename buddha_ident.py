from train_func import train_func
from prettytable import PrettyTable
import warnings
import argparse

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--model', type=str, default='AUNet')      # FCN_R50  FCN_R101  AUNet
parser.add_argument('--resume', type=str, default='False')
parser.add_argument('--buddha_data_path', type=str, default='/DISK0/DATA base/Buddhas/OneDrive-2024-08-29/')     # './archive/buddha_list'
args = parser.parse_args()

model_name = args.model
resume = args.resume
buddha_data_path = args.buddha_data_path
base_lr = 0.001
batch_size = 1
train_epoch = 7
folder_num = 10
if resume:
    reset_lr_epoch = 2  # 5
else:
    reset_lr_epoch = 2

t_correct = 0
t_count = 0
for f in range(folder_num):         #  + abs(f//2 - 1)
    count, correct = train_func(model_name=model_name, resume=resume, base_lr=base_lr, batch_size=batch_size,
                                              reset_lr_epoch=reset_lr_epoch, train_epoch=train_epoch,
                                              folder_num=folder_num, folder=f, buddha_data_path=buddha_data_path)
    t_correct += correct
    t_count += count

print('%s folders cross validation results:   correct/test count: %s/%s  AVE Precision: %.3f ' %
      (folder_num, t_correct, t_count, 100. * t_correct / t_count))
