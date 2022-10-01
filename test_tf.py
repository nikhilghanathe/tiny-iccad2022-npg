import argparse, os
import Utils
from Utils import custom_generator_train
import tensorflow as tf
import numpy as np


def evaluate():
	model_save_name = args.model_save_name
	#load test data
	if not os.path.exists('test_data.npy'):
		raise Utils.DataNotFound("RUN prepare_dataset.py before running this test script !")
	test_data = np.load('test_data.npy')
	test_labels = np.load('test_labels.npy')
	test_labels = tf.keras.utils.to_categorical(test_labels)

	#load model
	model = tf.keras.models.load_model(model_save_name)	
	model.summary()
	model.compile(metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), tf.keras.metrics.AUC()])
	test_metrics = model.evaluate(x=test_data, y=test_labels, batch_size=32, verbose=1, return_dict=True)
	print(test_metrics)



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--path_data', type=str, default='./tinyml_contest_data_training/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices')
    argparser.add_argument('--model_save_name', type=str, default='trained_dscnn_ref')
    args = argparser.parse_args()

    evaluate()
