import numpy as np
import os
import shutil
import torch
from torch.utils.data import DataLoader
import glob
from common_classes import load_data, run_test
from network import Net

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

save_images = 'restored_images'

shutil.rmtree(save_images, ignore_errors = True)
os.makedirs(save_images)
test_amplification_file = 'test_amplification.txt'
shutil.rmtree(test_amplification_file, ignore_errors = True)

test_files = glob.glob('SID/Sony/short/1*_00_*.ARW') 
gt_files = []
for x in test_files:
    gt_files = gt_files+ glob.glob('SID/Sony/long/*'+x[-17:-12]+'*.ARW')
dataloader_test = DataLoader(load_data(test_files,gt_files,test_amplification_file,2,gt_amp=True,training=False), batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

model = Net()
print('\n Network parameters : {}\n'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
model = model.to(device)
print('Device on GPU: {}'.format(next(model.parameters()).is_cuda))
#checkpoint = torch.load('demo_imgs/weights', map_location=device)
checkpoint = torch.load('weights/weights_2', map_location=device)
model.load_state_dict(checkpoint['model'])
mode = 'w'
metric_average_file = 'metric_average.txt'
save_csv_files = 'csv_files'
shutil.rmtree(save_csv_files, ignore_errors = True)
os.makedirs(save_csv_files)
iter_num = 500002
run_test(model, dataloader_test, iter_num, save_images, save_csv_files, metric_average_file, mode, training=False)
print('Restored images saved')
