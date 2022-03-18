import torch
from torch.utils.data import DataLoader
import glob
from common_classes import load_data, run_test
import time

dry_run = True  # If you wish to first test the entire workflow, for couple of iterations, make this TRUE
start_time = time.time()

opt = {'base_lr': 1e-4}  # Initial learning rate
opt['reduce_lr_by'] = 0.1  # Reduce learning rate by 10 times
opt['atWhichReduce'] = [500000]  # Reduce learning rate at these iterations.
opt['batch_size'] = 8
opt['atWhichSave'] = [2, 100002, 150002, 200002, 250002, 300002, 350002, 400002, 450002, 500002, 550000, 600000, 650002,
                      700002, 750000, 800000, 850002, 900002, 950000,
                      1000000]  # testing will be done at these iterations and corresponding model weights will be saved.
opt['iterations'] = 1000005  # The model will run for these many iterations.

dry_run_iterations = 100  # If dry run flag is set TRUE the code will terminate after these many iterations
metric_average_file = 'metric_average.txt'  # Average metrics will be saved here. Please note these are only for supervison. We used MATLAB for final PSNR and SSIM evaluation.
test_amplification_file = 'test_amplification.txt'  # Intermediate details for the test images, such as estimated amplification will be saved here.
train_amplification_file = 'train_amplification.txt'  # Intermediate details for the train images, such as estimated amplification will be saved here.

# These are folders
save_weights = 'weights'  # Model weights will be saved here.
save_images = 'images'  # Restored images will be saved here.
save_csv_files = 'csv_files'  # Other details such as loss value and learning rate will be saved in this file.

train_files = glob.glob('SID_cvpr_18_dataset/Sony/short/0*_00_0.1s.ARW')
train_files += glob.glob('SID_cvpr_18_dataset/Sony/short/2*_00_0.1s.ARW')


if dry_run:
    train_files = train_files[:1000]
    opt['iterations'] = dry_run_iterations

gt_files = []
for x in train_files:
    gt_files += glob.glob('SID_cvpr_18_dataset/Sony/long/*' + x[-17:-12] + '*.ARW')

print("Finished requisites in %s seconds" % (time.time() - start_time))
start_time = time.time()

dataloader_train = DataLoader(
    load_data(train_files, gt_files, train_amplification_file, 20, gt_amp=True, training=True),
    batch_size=opt['batch_size'], shuffle=True, num_workers=0, pin_memory=True)

print("Finished loading train files in %s seconds" % (time.time() - start_time))
start_time = time.time()

torch.save(dataloader_train, 'dataloader_train.pth')
del dataloader_train

print("Finished saving train files in %s seconds" % (time.time() - start_time))
start_time = time.time()


test_files = glob.glob('SID_cvpr_18_dataset/Sony/short/1*_00_0.1s.ARW')
if dry_run:
    test_files = test_files[:2]

gt_files = []
for x in test_files:
    gt_files = gt_files + glob.glob('SID_cvpr_18_dataset/Sony/long/*' + x[-17:-12] + '*.ARW')

dataloader_test = DataLoader(load_data(test_files, gt_files, test_amplification_file, 2, gt_amp=True, training=False),
                             batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

print("Finished loading test files in %s seconds" % (time.time() - start_time))
start_time = time.time()

torch.save(dataloader_test, 'dataloader_test.pth')
del dataloader_test
print("Finished saving test files in %s seconds" % (time.time() - start_time))

