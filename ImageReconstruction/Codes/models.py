import tensorflow as tf
import numpy as np
import reconstruction_net
import vgg19



def edge_extraction(gen_frames):
    # calculate the loss for each scale
    # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.

    channels = gen_frames.get_shape().as_list()[-1]
    pos = tf.constant(np.identity(channels), dtype=tf.float32)     # 3 x 3
    neg = -1 * pos
    filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)  # [-1, 1]
    print(filter_x.shape)
    filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]
    strides = [1, 1, 1, 1]  # stride of (1, 1)
    padding = 'SAME'

    gen_dx = tf.abs(tf.nn.conv2d(gen_frames, filter_x, strides, padding=padding))
    gen_dy = tf.abs(tf.nn.conv2d(gen_frames, filter_y, strides, padding=padding))
    
    edge = gen_dx ** 1 + gen_dy ** 1
    edge_clip  = tf.clip_by_value(edge, 0, 1)
    # condense into one tensor and avg
    return edge_clip

def seammask_extraction(mask):
    seam_mask = edge_extraction(tf.expand_dims(tf.reduce_mean(mask, axis=3),[3]))
    filters = tf.Variable(tf.constant([[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]],
                                 shape=[3,3,1,1]))
    test_conv1 =tf.nn.conv2d(seam_mask,filter=filters,strides=[1,1,1,1],padding="SAME")
    test_conv1 = tf.clip_by_value(test_conv1, 0, 1)
    test_conv2 =tf.nn.conv2d(test_conv1,filter=filters,strides=[1,1,1,1],padding="SAME")
    test_conv2 = tf.clip_by_value(test_conv2, 0, 1)
    test_conv3 =tf.nn.conv2d(test_conv2,filter=filters,strides=[1,1,1,1],padding="SAME")
    test_conv3 = tf.clip_by_value(test_conv3, 0, 1)
    # condense into one tensor and avg
    return test_conv3


def reconstruction(inputs):
    lr_stitched, hr_stitched = reconstruction_net.ReconstructionNet(inputs)
    
    return lr_stitched, hr_stitched

    


def Vgg19_simple_api(rgb, reuse):
    return vgg19.Vgg19_simple(rgb, reuse)
    

