import argparse, os
import Utils
from Utils import custom_generator_train
import tensorflow as tf
import models.keras_model as keras_model
import numpy as np


#define data generator
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    horizontal_flip=False,
    #brightness_range=(0.9, 1.2),
    #contrast_range=(0.9, 1.2),
    validation_split=0.2
)



#custom callback to set weights of convolution from ee-1 to conv just before ee-final classification
class customCallback(tf.keras.callbacks.Callback):
  def __init__(self, num_epochs, train_count):
    self. epoch_threshold_max = (num_epochs//5) *4
    self.train_count = train_count
    self.no_transfer = False

  #distill early-exit knowledge and copy it at final exit
  def on_train_batch_begin(self, batch, logs=None):
    if not self.no_transfer:
        for layer in self.model.layers:
            if layer.name=='dw_ee_1':
                conv_layer = layer
            if layer.name=='dw_ee_f':
                depthconv_eefinal_layer = layer

        weights = conv_layer.get_weights()
        depthconv_eefinal_layer.set_weights(weights)     
  
  def on_epoch_end(self, epoch, logs=None):
        if self.train_count==0 and epoch==self.epoch_threshold_max:
            self.no_transfer = True
        # if self.train_count==1 and epoch==5:
        #     self.no_transfer = True
        # if self.train_count==2 and epoch==15:
        #     self.no_transfer = True




def main():
    # Hyperparameters
    BATCH_SIZE = args.batchsz
    BATCH_SIZE_TEST = args.batchsz
    LR = args.lr
    EPOCHS = args.epoch
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices
    model_save_name= args.model_save_name
    model_arch = args.model_arch


    # Start dataset loading - 
    #-- !! RUN prepare_dataset.py before running the train_script !!
    # fname = 'train_data.npy'
    if not os.path.exists('train_data.npy'):
        raise Utils.DataNotFound("RUN prepare_dataset.py before running this train_script !")

    train_data = np.load('train_data.npy')
    train_labels = np.load('train_labels.npy')
    train_labels = tf.keras.utils.to_categorical(train_labels)
     

    print("Training Dataset loading finish.")

    #Uncomment if not using pretrained model
    # if model_arch=='ds_cnn_ev':
    #     new_model = keras_model.ds_cnn_ev(alpha=1)
    # else:
    #     new_model = keras_model.ds_cnn(alpha=1)
    

    #load pretrained model
    new_model = tf.keras.models.load_model('saved_models/pretrained_trained_dscnn_ev_ref')
    new_model.summary()

    # from keras_flops import get_flops
    # # calc FLOPS
    # flops = get_flops(new_model, batch_size=BATCH_SIZE)
    # print(f"FLOPS: {flops / 10 ** 9:.03} G")
    

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(train_data)
    train_gen = datagen.flow( train_data, train_labels,  batch_size=BATCH_SIZE)
    val_gen =None

    #train model
    new_model = train_epochs(new_model, train_gen, val_gen, EPOCHS, BATCH_SIZE, 0.0001, model_save_name, 0, len(train_data))



def train_epochs(new_model, train_gen, val_gen, epoch_count, batch_size,
                 learning_rate, model_save_name, train_count, train_data_len):

    #compile the model
    #new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy', metrics='accuracy', loss_weights=None, weighted_metrics=None, run_eagerly=False)
    # fits the model on batches with real-time data augmentation:
    print('----------------------------------------------------')
    print('STARTING TRAINING....')
    print('----------------------------------------------------')
    
    gen = custom_generator_train(train_gen)
    History = new_model.fit( gen, steps_per_epoch=((train_data_len)) / batch_size, epochs=epoch_count, callbacks=[customCallback(epoch_count, train_count)])

    print('DONE!')  
    print('----------------------------------------------------')
    print('Saving Final Model....')
    print('----------------------------------------------------')
    new_model.save("saved_models/" + model_save_name+"_ee.h5")
    print('DONE!')



    #strip away the final exit and save the compact early-exit model
    cnt=0
    for layer in new_model.layers:
        if layer.name =='dense_2_ee_1':
            ee_num = cnt
        cnt +=1
    model = tf.keras.models.Model(inputs=new_model.inputs[0], outputs=new_model.layers[ee_num].output)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='categorical_crossentropy', metrics='accuracy', loss_weights=None,
         weighted_metrics=None, run_eagerly=False)
    model.save("saved_models/" + model_save_name+'.h5')
    model.save("saved_models/" + model_save_name)


    #load test data
    # test_data = np.load('test_data.npy')
    # test_labels = np.load('test_labels.npy')
    # test_labels = tf.keras.utils.to_categorical(test_labels)
    # test_metrics = new_model.evaluate(x=(test_data, test_labels), y=test_labels, batch_size=batch_size, verbose=1, return_dict=True)
    # print(test_metrics)

    return new_model

    print('Finish training')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=40)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
    argparser.add_argument('--batchsz', type=int, help='total batchsz for traindb', default=32)
    argparser.add_argument('--cuda', type=int, default=0)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='./tinyml_contest_data_training/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices')
    argparser.add_argument('--model_save_name', type=str, default='trained_dscnn')
    argparser.add_argument('--model_arch', type=str, default='ds_cnn_ev')
    args = argparser.parse_args()

    main()
