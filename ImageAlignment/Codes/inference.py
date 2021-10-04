import tensorflow as tf
import os
import numpy as np
import cv2 as cv

from models import H_estimator
from utils import DataLoader, load, save
import constant
import skimage

slim = tf.contrib.slim

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU
test_folder = constant.TEST_FOLDER
snapshot_dir =  constant.SNAPSHOT_DIR + '/model.ckpt-1000000'
batch_size = constant.TEST_BATCH_SIZE

# define dataset
with tf.name_scope('dataset'):
    ##########testing###############
    
    test_inputs = tf.placeholder(shape=[batch_size, 128, 128, 3 * 2], dtype=tf.float32)
    print('test inputs = {}'.format(test_inputs))



with tf.variable_scope('generator', reuse=None):
    print('testing = {}'.format(tf.get_variable_scope().name))
    test_net1_f, test_net2_f, test_net3_f, test_warp2_H1, test_warp2_H2, test_warp2_H3, test_one_warp_H1, test_one_warp_H2, test_one_warp_H3 = H_estimator(test_inputs, test_inputs, False)
   


config = tf.ConfigProto()
config.gpu_options.allow_growth = True      
with tf.Session(config=config) as sess:
    # dataset
    data_loader = DataLoader(test_folder)

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
        length = 1106
        psnr_list = []
        ssim_list = []

        for i in range(0, length):
            #load test data
            input_clip = np.expand_dims(data_loader.get_data_clips(i, 128, 128), axis=0)
            
            # inference
            _, _, _, _, _, warp, _, _, warp_one = sess.run([test_net1_f, test_net2_f, test_net3_f, test_warp2_H1, test_warp2_H2, test_warp2_H3, test_one_warp_H1, test_one_warp_H2, test_one_warp_H3], feed_dict={test_inputs: input_clip})
            
            
            warp = (warp+1) * 127.5    
            warp = warp[0] 
            warp_one = warp_one[0]
            input1 = (input_clip[...,0:3]+1) * 127.5    
            input1 = input1[0]
            input2 = (input_clip[...,3:6]+1) * 127.5    
            input2 = input2[0]
            
            # compute psnr/ssim
            psnr = skimage.measure.compare_psnr(input1*warp_one, warp*warp_one, 255)
            ssim = skimage.measure.compare_ssim(input1*warp_one, warp*warp_one, data_range=255, multichannel=True)

            
            print('i = {} / {}, psnr = {:.6f}'.format( i+1, length, psnr))
            
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            
            
        print("===================Results Analysis==================")   
        psnr_list.sort(reverse = True)
        psnr_list_30 = psnr_list[0 : 331]
        psnr_list_60 = psnr_list[331: 663]
        psnr_list_100 = psnr_list[663: -1]
        print("top 30%", np.mean(psnr_list_30))
        print("top 30~60%", np.mean(psnr_list_60))
        print("top 60~100%", np.mean(psnr_list_100))
        print('average psnr:', np.mean(psnr_list))
        
        ssim_list.sort(reverse = True)
        ssim_list_30 = ssim_list[0 : 331]
        ssim_list_60 = ssim_list[331: 663]
        ssim_list_100 = ssim_list[663: -1]
        print("top 30%", np.mean(ssim_list_30))
        print("top 30~60%", np.mean(ssim_list_60))
        print("top 60~100%", np.mean(ssim_list_100))
        print('average ssim:', np.mean(ssim_list))

    inference_func(snapshot_dir)
    

