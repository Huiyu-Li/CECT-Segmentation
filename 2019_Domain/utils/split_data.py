import csv
import math
import numpy as np
import os
import sys
import shutil
import random
import re
from glob import glob

# split data
import re
def atoi(s):
	return int(s) if s.isdigit() else s

def natural_keys(text):
	return [atoi(c) for c in re.split('(\d+)', text)]

def write_csv(file,valid_lists):
	w = csv.writer(file)
	for name in valid_lists:
		tomo_name = name
		seg_name = tomo_name.replace('subtomogram_mrc', 'processed_densitymap_mrc_mrc2binary')
		seg_name = seg_name.replace('tomotarget', 'packtarget')
		no = re.findall(r"\d+", tomo_name.split('/')[-1])
		assert len(no) == 1, 'len(no) ==1'
		class_label = int(no[0]) // 500
		# print(no, class_label)
		w.writerow((tomo_name, seg_name, class_label))  # attention: the first row defult to tile

def Split_data(data_dir,csv_dir):
	#Split data into train-valid-test = [6:1:3]= [0.6,0.1,0.3]
	# NewData\data1_SNR003\subtomogram_mrc\tomotarget0.mrc
	# NewData\data1_SNR003\processed_densitymap_mrc_mrc2binary\packtarget0.mrc

	sub_file = ['data1_SNR003','data2_SNR005','data3_SNRinfinity']#['data1_SNR003']#
	tomo_list = []

	for sub in sub_file:
		file_list = glob(os.path.join(data_dir,sub,'subtomogram_mrc','*.mrc'))
		tomo_list.extend(file_list)

	random.seed(9)
	random.shuffle(tomo_list)
	total = len(tomo_list)
	train_lists = tomo_list[0:int(total*0.6)]
	valid_lists = tomo_list[int(total*0.6):int(total*0.7)]
	test_lists = tomo_list[int(total*0.7):total]
	print(total,len(train_lists),len(valid_lists),len(test_lists))

	# clear the exists file
	if os.path.isdir(csv_dir):
		shutil.rmtree(csv_dir)
	os.mkdir(csv_dir)

	with open(os.path.join(csv_dir, 'train.csv'), 'w') as file:
		write_csv(file,train_lists)
	with open(os.path.join(csv_dir, 'valid.csv'), 'w') as file:
		write_csv(file,valid_lists)
	with open(os.path.join(csv_dir, 'test.csv'), 'w') as file:
		write_csv(file,test_lists)

if __name__ == '__main__':
	data_dir = '/media/lihuiyu/NewData/'
	csv_dir = './TVTcsv'
	Split_data(data_dir,csv_dir)
