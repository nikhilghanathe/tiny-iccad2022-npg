import tensorflow as tf
# import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

# import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization, Dropout, Reshape, Softmax
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2



#define endpoint layer for auxilliary loss calculation
class Endpoint_ee1(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.batch_size = 32
        

    @tf.function
    def loss_fn(self, ee_1, ee_final, targets):
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        W_aux = 0.5
        P_aux = 1.0
        num_classes=2

        self.batch_size = targets.shape[0]
        y_true = targets
        y_true_transformed_ee1, y_true_transformed_eefinal = [], []
        y_pred_ee1_transformed = []
        y_pred_eefinal_transformed = []
        

        y_pred_ee1 = ee_1
        y_pred_eefinal = ee_final

        if self.batch_size==None:
            self.batch_size=32
        loss_ee1, loss_eefinal =0.0, 0.0

        #for EE-1
        for i in range(0, self.batch_size):
            arg_max_true = tf.keras.backend.argmax(y_true[i])
            arg_max_true = tf.cast(arg_max_true, dtype='int32')
            arg_max_true = tf.cast(arg_max_true, dtype='int32')
            prob_list = y_pred_ee1[i]
            values, indices =  tf.math.top_k(prob_list, k=1)
            
            [score_max_1] = tf.split(values, num_or_size_splits=1)
            [arg_max_1] = tf.split(indices, num_or_size_splits=1)
            
            arg_max_true = tf.reshape(arg_max_true, [1])
            if tf.math.equal(arg_max_true, arg_max_1):
              if True:
                y_uncrtn_neg = tf.one_hot([arg_max_true], depth=num_classes, on_value=1., off_value=0.0, dtype='float32')
                y_uncrtn = y_uncrtn_neg
              else:
                y_uncrtn_pos = tf.one_hot([arg_max_true], depth=num_classes, on_value=P_aux, off_value=0.0, dtype='float32')
                y_uncrtn = y_uncrtn_pos
            else:
                y_uncrtn = tf.one_hot([arg_max_true], depth=num_classes, on_value=P_aux, off_value=0.0, dtype='float32')
            y_true_transformed_ee1.append(y_uncrtn)
        y_true_transformed_ee1 = tf.reshape(y_true_transformed_ee1, [self.batch_size,num_classes])
        
        loss_cce =  cce(y_true_transformed_ee1, y_pred_ee1) 
        #multiply Loss with W_aux (0<W_aux<1) value and return
        return tf.multiply(W_aux, loss_cce)

    def call(self, ee_1, ee_final, targets=None, sample_weight=None):
        if targets is not None:
            loss = self.loss_fn(ee_1, ee_final,  targets)
            self.add_loss(loss)
            self.add_metric(loss, name='aux_loss', aggregation='mean')
        return ee_1, ee_final

    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)



#depthwise-separable model
def ds_cnn(alpha):
    input_shape = [1250,1,1]
    filters = 4
    weight_decay = 1e-4
    regularizer = l2(weight_decay)
    final_pool_size = (6,1)
    num_classes = 2
    
    # Model layers
    # Input pure conv2d
    inputs = Input(shape=input_shape)
    x = Conv2D(filters, name = 'conv_1', kernel_size=(10,3), strides=(2,1), padding='same', kernel_regularizer=regularizer)(inputs)
    x = BatchNormalization(name = 'bn_1')(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.2)(x)


    filters=int(8*alpha)
    # First layer of separable depthwise conv2d
    # Separable consists of depthwise conv2d followed by conv2d with 1x1 kernels
    x = DepthwiseConv2D(name = 'dw_1',depth_multiplier=1, strides=2, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, name = 'pw_1',kernel_size=(1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    filters=int(8*alpha)
    # Second layer of separable depthwise conv2d
    x = DepthwiseConv2D(name = 'dw_2',depth_multiplier=1,  strides=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, name = 'pw_2',kernel_size=(1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    filters=int(8*alpha)
    # Third layer of separable depthwise conv2d
    x = DepthwiseConv2D(name = 'dw_3',depth_multiplier=1, strides=2,kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, name = 'pw_3',kernel_size=(1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    filters=int(8*alpha)
    # Fourth layer of separable depthwise conv2d
    x = DepthwiseConv2D(name = 'dw_4',depth_multiplier=1, strides=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters,name = 'pw_4', kernel_size=(1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    filters=int(16*alpha)
    # Fourth layer of separable depthwise conv2d
    x = DepthwiseConv2D(name = 'dw_5',depth_multiplier=1, strides=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters,name = 'pw_5', kernel_size=(1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Reduce size and apply final softmax
    x = Dropout(rate=0.2)(x)

    x = AveragePooling2D(pool_size=final_pool_size)(x)
    x = Flatten()(x)

    x = Dense(num_classes*5, activation='softmax')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)   

    return model


#model with early-exit
def ds_cnn_ev(alpha):
    input_shape = [1250,1,1]
    filters = 4
    weight_decay = 1e-4
    regularizer = l2(weight_decay)
    final_pool_size = (6,1)
    num_classes = 2
    
    # Model layers
    # Input pure conv2d
    inputs = Input(shape=input_shape)
    x = Conv2D(filters, name = 'conv_1', kernel_size=(10,3), strides=(2,1), padding='same', kernel_regularizer=regularizer)(inputs)
    x = BatchNormalization(name = 'bn_1')(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.2)(x)


    filters=int(8*alpha)
    # First layer of separable depthwise conv2d
    # Separable consists of depthwise conv2d followed by conv2d with 1x1 kernels
    x = DepthwiseConv2D(name = 'dw_1',depth_multiplier=1, strides=2, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, name = 'pw_1',kernel_size=(1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    filters=int(8*alpha)
    # Second layer of separable depthwise conv2d
    x = DepthwiseConv2D(name = 'dw_2',depth_multiplier=1,  strides=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, name = 'pw_2',kernel_size=(1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x_ee = x

    filters=int(8*alpha)
    # Third layer of separable depthwise conv2d
    x = DepthwiseConv2D(name = 'dw_3',depth_multiplier=1, strides=2,kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, name = 'pw_3',kernel_size=(1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    filters=int(8*alpha)
    # Fourth layer of separable depthwise conv2d
    x = DepthwiseConv2D(name = 'dw_4',depth_multiplier=1, strides=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters,name = 'pw_4', kernel_size=(1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    filters=int(16*alpha)
    # Fourth layer of separable depthwise conv2d
    x = DepthwiseConv2D(name = 'dw_5',depth_multiplier=1, strides=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters,name = 'pw_5', kernel_size=(1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Reduce size and apply final softmax
    x = Dropout(rate=0.2)(x)


    #add EE
    #EE
    # x_ee_1 = Conv2D(8,
    #              kernel_size=1,
    #              strides=1,
    #              padding='same',
    #              kernel_initializer='he_normal',
    #              kernel_regularizer=l2(1e-4), name='pw_ee_1')(x_ee)

    # depthwise conv without batch norm and depth_multiplier=1
    conv_concat_fmaps = tf.keras.layers.DepthwiseConv2D(name='dw_ee_1',
                   kernel_size=[2,2],
                   strides=2,
                   padding='same',
                   depth_multiplier=1,
                   activation='relu',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-4))(x_ee)

    conv_ee1_fmaps = AveragePooling2D(pool_size=final_pool_size)(conv_concat_fmaps)
    y_ee_1 = Flatten()(conv_ee1_fmaps)
    ee_1 = Dense(10, name='dense_1_ee_1',
                    activation='softmax',
                    kernel_initializer='he_normal')(y_ee_1)
    ee_1 = Dense(num_classes, name='dense_2_ee_1',
                    activation='softmax',
                    kernel_initializer='he_normal')(ee_1)
    


    #EV-assisted classification
    depth_conv_eefinal_out = tf.keras.layers.DepthwiseConv2D(name='dw_ee_f',
                   kernel_size=[2,2],
                   strides=1,
                   padding='same',
                   depth_multiplier=1,
                   activation='relu',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-4))(x[:,:,:,:8])


    x = AveragePooling2D(pool_size=final_pool_size)(x)
    y = Flatten()(x)
    x_ee_pooled = AveragePooling2D(pool_size=final_pool_size)(depth_conv_eefinal_out[:,:,:,:8])
    y_depthconv_eefinal = Flatten()(x_ee_pooled)
    y_combined = tf.keras.layers.concatenate([y, y_depthconv_eefinal])                   



    # #final classification   
    # x_pooled = AveragePooling2D(pool_size=final_pool_size)(x)
    # y = Flatten()(x_pooled)
    x = Dense(num_classes*5, activation=None)(y_combined)
    outputs = Dense(num_classes, activation='softmax')(x)
    

    #add endpoint layer
    targets = Input(shape=[num_classes], name='input_2')
    ee_1, outputs = Endpoint_ee1(name='endpoint')(ee_1, outputs, targets)


    # Instantiate model.
    model = Model(inputs=[inputs, targets], outputs=[ee_1, outputs])   

    return model




