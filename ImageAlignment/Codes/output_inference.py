import tensorflow as tf
import os
import numpy as np
import cv2


from models import H_estimator, output_H_estimator
from utils import DataLoader, load, save
import constant
import skimage


os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU
train_folder = constant.TRAIN_FOLDER
test_folder = constant.TEST_FOLDER
snapshot_dir =  constant.SNAPSHOT_DIR + '/model.ckpt-1000000'
batch_size = constant.TEST_BATCH_SIZE

# define dataset
with tf.name_scope('dataset'):
    ##########testing###############
    test_inputs = tf.placeholder(shape=[batch_size, None, None, 3 * 2], dtype=tf.float32)
    test_size = tf.placeholder(shape=[batch_size, 2, 1], dtype=tf.float32)
    print('test inputs = {}'.format(test_inputs))
    print('test size = {}'.format(test_size))



with tf.variable_scope('generator', reuse=None):
    print('testing = {}'.format(tf.get_variable_scope().name))
    test_coarsealignment = output_H_estimator(test_inputs, test_size, False)
    


config = tf.ConfigProto()
config.gpu_options.allow_growth = True      
with tf.Session(config=config) as sess:


    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init global successfully!')

    # tf saver
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)

    restore_var = [v for v in tf.global_variables()]
    loader = tf.train.Saver(var_list=restore_var)

    def inference_func(ckpt):
        print("============")
        print(ckpt)
        load(loader, sess, ckpt)
        print("============")
        
        print("------------------------------------------")
        print("generating aligned images for training set")
        # dataset
        data_loader = DataLoader(train_folder)
        length = 100
        for i in range(0, length):
            input_clip = np.expand_dims(data_loader.get_data_clips(i, None, None), axis=0)
            size_clip = np.expand_dims(data_loader.get_size_clips(i), axis=0)
            
            coarsealignment = sess.run(test_coarsealignment, feed_dict={test_inputs: input_clip, test_size: size_clip})
            
            coarsealignment = coarsealignment[0]
            warp1 = (coarsealignment[...,0:3]+1.)*127.5
            warp2 = (coarsealignment[...,3:6]+1.)*127.5
            mask1 = coarsealignment[...,6:9] * 255
            mask2 = coarsealignment[...,9:12] * 255
            
            path1 = '../output/training/warp1/' + str(i+1).zfill(6) + ".jpg"
            cv2.imwrite(path1, warp1)
            path2 = '../output/training/warp2/' + str(i+1).zfill(6) + ".jpg"
            cv2.imwrite(path2, warp2)
            path3 = '../output/training/mask1/' + str(i+1).zfill(6) + ".jpg"
            cv2.imwrite(path3, mask1)
            path4 = '../output/training/mask2/' + str(i+1).zfill(6) + ".jpg"
            cv2.imwrite(path4, mask2)
                   
            print('i = {} / {}'.format(i+1, length))

        print("-----------training set done--------------")
        print("------------------------------------------")
        
        print()
        print()
        
        print("------------------------------------------")
        print("generating aligned images for testing set")
        # dataset
        data_loader = DataLoader(test_folder)
        length = 1106
        for i in range(0, length):
            input_clip = np.expand_dims(data_loader.get_data_clips(i, None, None), axis=0)
            size_clip = np.expand_dims(data_loader.get_size_clips(i), axis=0)
            
            coarsealignment = sess.run(test_coarsealignment, feed_dict={test_inputs: input_clip, test_size: size_clip})
            
            coarsealignment = coarsealignment[0]
            warp1 = (coarsealignment[...,0:3]+1.)*127.5
            warp2 = (coarsealignment[...,3:6]+1.)*127.5
            mask1 = coarsealignment[...,6:9] * 255
            mask2 = coarsealignment[...,9:12] * 255
            
            path1 = '../output/testing/warp1/' + str(i+1).zfill(6) + ".jpg"
            cv2.imwrite(path1, warp1)
            path2 = '../output/testing/warp2/' + str(i+1).zfill(6) + ".jpg"
            cv2.imwrite(path2, warp2)
            path3 = '../output/testing/mask1/' + str(i+1).zfill(6) + ".jpg"
            cv2.imwrite(path3, mask1)
            path4 = '../output/testing/mask2/' + str(i+1).zfill(6) + ".jpg"
            cv2.imwrite(path4, mask2)
                     
            print('i = {} / {}'.format(i+1, length))

        print("-----------testing set done--------------")
        print("------------------------------------------")

     
    inference_func(snapshot_dir)



