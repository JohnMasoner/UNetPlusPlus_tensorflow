import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import os
from tensorflow.contrib.data import Dataset
from tensorflow.contrib.layers import conv2d


reg = slim.l2_regularizer(scale=0.001) 
 
def crop_and_concat(x1,x2):
    with tf.name_scope("crop_and_concat"):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 3)


def standard_unit(inputs, stage, nb_filter, kernel_size=3):
    x = slim.conv2d(inputs, nb_filter, [3, 3], rate=1,activation_fn=None, weights_regularizer=reg)
    x = slim.batch_norm(x)
    x = tf.nn.relu(x)
    #x = slim.dropout(x)
    x = slim.conv2d(x, nb_filter, [3, 3], rate=1, scope=stage,activation_fn=None,weights_regularizer=reg)
    x = slim.batch_norm(x)
    x = tf.nn.relu(x)
    #x = slim.dropout(x)
    return x
 
def upsample(x,num_outputs,batch_size=4):
    pool_size = 2
    input_wh=int(x.shape[1])
    in_channels=int(x.shape[-1])
    output_shape=(batch_size,input_wh*2,input_wh*2,num_outputs)
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, num_outputs, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x, deconv_filter, output_shape, strides=[1, pool_size, pool_size, 1])
    return deconv
 
 
def UNet_pp(inputs, reg,deep_supervision=True):  # Unet
    '''
     1-1---> 1-2 ---> 1-3 ---> 1-4 --->1-5
        \   /   \   /    \    /   \   /
         2-1 --->2-2 ---> 2-3 --->2-4
           \    /   \    /   \   /
            3-1 ---> 3-2 ---> 3-3
              \     /   \    / 
                4-1---> 4-2
                  \     /
                    5-1  
    '''
 
    nb_filter = [32,64,128,256,512]
 
    conv1_1 = standard_unit(inputs,stage='stage_11',nb_filter=nb_filter[0])
    pool1 = slim.max_pool2d(conv1_1, [2, 2], padding='SAME')
 
    conv2_1 = standard_unit(pool1,stage='stage_21',nb_filter=nb_filter[1])
    pool2 = slim.max_pool2d(conv2_1, [2, 2], padding='SAME')
 
    conv3_1 = standard_unit(pool2,stage='stage_31',nb_filter=nb_filter[2])
    pool3 = slim.max_pool2d(conv3_1, [2, 2], padding='SAME')
 
    conv4_1 = standard_unit(pool3,stage='stage_41',nb_filter=nb_filter[3])
    pool4 = slim.max_pool2d(conv4_1, [2, 2], padding='SAME')
 
    conv5_1 = standard_unit(pool4,stage='stage_51',nb_filter=nb_filter[4])
 
    up1_2 = upsample(conv2_1,num_outputs=nb_filter[0])
    #up1_2 = slim.conv2d_transpose(conv2_1,num_outputs=nb_filter[0],kernel_size=2,stride=2)
    conv1_2 = tf.concat([conv1_1,up1_2],3)
    #conv1_2 = crop_and_concat(conv1_1,up1_2)
    #conv1_2 = np.concatenate((conv1_1,up1_2),3)
    conv1_2 = standard_unit(conv1_2,stage='stage_12',nb_filter=nb_filter[0])
 
    up2_2 = upsample(conv3_1,num_outputs=nb_filter[1])
    #up2_2 = slim.conv2d_transpose(conv3_1,num_outputs=nb_filter[1],kernel_size=2,stride=2)
    conv2_2 = tf.concat([conv2_1,up2_2],3)
    conv2_2 = standard_unit(conv2_2,stage='stage_22',nb_filter=nb_filter[1])
 
    up3_2 = upsample(conv4_1,num_outputs=nb_filter[2])
    #up3_2 = slim.conv2d_transpose(conv4_1,num_outputs=nb_filter[2],kernel_size=2,stride=2)
    conv3_2 = tf.concat([conv3_1,up3_2],3)
    conv3_2 = standard_unit(conv3_2,stage='stage_32',nb_filter=nb_filter[2])
 
    up4_2 = upsample(conv5_1,num_outputs=nb_filter[3])
    #up4_2 = slim.conv2d_transpose(conv5_1,num_outputs=nb_filter[3],kernel_size=2,stride=2)
    conv4_2 = tf.concat([conv4_1,up4_2],3)
    conv4_2 = standard_unit(conv4_2,stage='stage_42',nb_filter=nb_filter[3])
 
    up1_3 = upsample(conv2_2,num_outputs=nb_filter[0])
    #up1_3 = slim.conv2d_transpose(conv2_2,num_outputs=nb_filter[0],kernel_size=2,stride=2)
    conv1_3 = tf.concat([conv1_1,conv1_2,up1_3],3)
    conv1_3 = standard_unit(conv1_3,stage='stage_13',nb_filter=nb_filter[0])
 
    up2_3 = upsample(conv3_2,num_outputs=nb_filter[1])
    #up2_3 = slim.conv2d_transpose(conv3_2,num_outputs=nb_filter[1],kernel_size=2,stride=2)
    conv2_3 = tf.concat([conv2_1,conv2_2,up2_3],3)
    conv2_3 = standard_unit(conv2_3,stage='stage_23',nb_filter=nb_filter[1])
 
    up3_3 = upsample(conv4_2,num_outputs=nb_filter[2])
    #up3_3 = slim.conv2d_transpose(conv4_2,num_outputs=nb_filter[2],kernel_size=2,stride=2)
    conv3_3 = tf.concat([conv3_1,conv3_2,up3_3],3)
    conv3_3 = standard_unit(conv3_3,stage='stage_33',nb_filter=nb_filter[2])
 
    up1_4 = upsample(conv2_3,num_outputs=nb_filter[0])
    #up1_4 = slim.conv2d_transpose(conv2_3,num_outputs=nb_filter[0],kernel_size=2,stride=2)
    conv1_4 = tf.concat([conv1_1,conv1_2,conv1_3,up1_4],3)
    conv1_4 = standard_unit(conv1_4,stage='stage_14',nb_filter=nb_filter[0])
 
    up2_4 = upsample(conv3_3,num_outputs=nb_filter[1])
    #up2_4 = slim.conv2d_transpose(conv3_3,num_outputs=nb_filter[1],kernel_size=2,stride=2)
    conv2_4 = tf.concat([conv2_1,conv2_2,conv2_3,up2_4],3)
    conv2_4 = standard_unit(conv2_4,stage='stage_24',nb_filter=nb_filter[1])
 
    up1_5 = upsample(conv2_4,num_outputs=nb_filter[2])
    #up1_5 = slim.conv2d_transpose(conv2_4,num_outputs=nb_filter[0],kernel_size=2,stride=2)
    conv1_5 = tf.concat([conv1_1,conv1_2,conv1_3,conv1_4,up1_5],3)
    conv1_5 = standard_unit(conv1_5,stage='stage_15',nb_filter=nb_filter[0])

    nestnet_output_1 = slim.conv2d(conv1_2, 1, [1, 1], rate=1, activation_fn=tf.nn.sigmoid, scope='output_1',weights_regularizer=slim.l2_regularizer(scale=0.0001))
    nestnet_output_2 = slim.conv2d(conv1_3, 1, [1, 1], rate=1, activation_fn=tf.nn.sigmoid, scope='output_2',weights_regularizer=slim.l2_regularizer(scale=0.0001))
    nestnet_output_3 = slim.conv2d(conv1_4, 1, [1, 1], rate=1, activation_fn=tf.nn.sigmoid, scope='output_3',weights_regularizer=slim.l2_regularizer(scale=0.0001))
    nestnet_output_4 = slim.conv2d(conv1_5, 1, [1, 1], rate=1, activation_fn=tf.nn.sigmoid, scope='output_4',weights_regularizer=slim.l2_regularizer(scale=0.0001))
    if deep_supervision:
        h_deconv_concat = tf.concat([nestnet_output_1, nestnet_output_2,nestnet_output_3,nestnet_output_4],3)
        h_deconv_concat = conv2d(inputs=h_deconv_concat,num_outputs = 3, kernel_size=3, activation_fn=None)
        h_deconv_concat = tf.tanh(h_deconv_concat)
        return h_deconv_concat
    else:
        return nestnet_output_4
