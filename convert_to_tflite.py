import os
import tensorflow as tf
assert tf.__version__.startswith('2')
import argparse
import numpy as np

def main():
    path_data = args.path_data
    model_save_name = args.model_save_name
    model = tf.keras.models.load_model(model_save_name)
    model.summary()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with tf.io.gfile.GFile('saved_models/trained_dscnn_float.tflite', 'wb') as float_file:
        float_file.write(tflite_model)
    

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--path_data', type=str, help='path to train dataset', default='tinyml_contest_data_training')
    argparser.add_argument('--path_indices', type=str, default='./data_indices')
    argparser.add_argument('--model_save_name', type=str, default='saved_models/trained_dscnn_ref')
    args = argparser.parse_args()
    main()