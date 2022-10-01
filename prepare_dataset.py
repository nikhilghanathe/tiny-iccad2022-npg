#prepare the dataset by converting into a numpy array
import argparse
import csv, os
import numpy as np
from Utils import loadCSV, txt_to_numpy

def prepare_data(root_dir, indice_dir, size, mode):
	csvdata = loadCSV(os.path.join(indice_dir, mode+'_indice.csv'))
	
	data, labels = [], []
	for i, (k, v) in enumerate(csvdata.items()):
		# names_list_train.append(str(k) + ' ' + str(v[0]))
		fname = str(k) + ' ' + str(v[0])
		text_path = os.path.join(root_dir,fname.split(' ')[0])
		if not os.path.isfile(text_path):
			print(text_path + 'does not exist')
			return None
		IEGM_seg = txt_to_numpy(text_path, size).reshape(size, 1, 1)
		label = int(fname.split(' ')[1])
		data.append(IEGM_seg)
		labels.append(label)
	data= np.array(data)
	labels = np.array(labels)
	with open(mode+'_data.npy', 'wb') as fp:
		np.save(fp, data)
	with open(mode+'_labels.npy', 'wb') as fp:
		np.save(fp, labels)




if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--path_data', type=str, help='path to train dataset', default='tinyml_contest_data_training')
    argparser.add_argument('--path_indices', type=str, default='./data_indices')
    argparser.add_argument('--size', type=int, default=1250)
    args = argparser.parse_args()
    path_to_data = args.path_data
    indice_dir = args.path_indices
    size = args.size
    prepare_data(path_to_data, indice_dir, size, mode='train')
    prepare_data(path_to_data, indice_dir, size, mode='test')

