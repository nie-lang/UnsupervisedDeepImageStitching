import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorDLT import solve_DLT
from tf_spatial_transform import transform
from tensorflow.contrib.layers import conv2d


def H_model(inputs_aug, inputs, is_training, patch_size=128.):

    batch_size = tf.shape(inputs)[0]
    net1_f, net2_f, net3_f = build_model(inputs_aug, is_training)
    
    
    M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
                  [0., patch_size / 2.0, patch_size / 2.0],
                  [0., 0., 1.]]).astype(np.float32)
    M_tensor = tf.constant(M, tf.float32)
    M_tile = tf.tile(tf.expand_dims(M_tensor, [0]), [batch_size, 1, 1])
    # Inverse of M
    M_inv = np.linalg.inv(M)
    M_tensor_inv = tf.constant(M_inv, tf.float32)
    M_tile_inv = tf.tile(tf.expand_dims(M_tensor_inv, [0]), [batch_size, 1, 1])
    
    H1 = solve_DLT(net1_f, patch_size)
    H2 = solve_DLT(net1_f+net2_f, patch_size)
    H3 = solve_DLT(net1_f+net2_f+net3_f, patch_size)
    
    H1_mat = tf.matmul(tf.matmul(M_tile_inv, H1), M_tile)
    H2_mat = tf.matmul(tf.matmul(M_tile_inv, H2), M_tile)
    H3_mat = tf.matmul(tf.matmul(M_tile_inv, H3), M_tile)
    
    image2_tensor = inputs[..., 3:6]
    warp2_H1 = transform(image2_tensor, H1_mat)
    warp2_H2 = transform(image2_tensor, H2_mat)
    warp2_H3 = transform(image2_tensor, H3_mat)
    
    one = tf.ones_like(image2_tensor, dtype=tf.float32)
    one_warp_H1 = transform(one, H1_mat)
    one_warp_H2 = transform(one, H2_mat)
    one_warp_H3 = transform(one, H3_mat)
    
    return net1_f, net2_f, net3_f, warp2_H1, warp2_H2, warp2_H3, one_warp_H1, one_warp_H2, one_warp_H3

def H_model_v2(inputs, is_training):
    net1_f, net2_f, net3_f = build_model(inputs, is_training)  
    shift = net1_f + net2_f + net3_f 
    
    return shift


def build_model(inputs, is_training):
    with tf.variable_scope('model'):
        input1 = inputs[...,0:3]
        input2 = inputs[...,3:6]
        #resize to 128*128
        input1 = tf.image.resize_images(input1, [128,128],method=0)
        input2 = tf.image.resize_images(input2, [128,128],method=0)
        input1 = tf.expand_dims(tf.reduce_mean(input1, axis=3),[3])
        input2 = tf.expand_dims(tf.reduce_mean(input2, axis=3),[3])
        net1_f, net2_f, net3_f = _H_model(input1, input2, is_training)
        return net1_f, net2_f, net3_f

def _conv_block(x, num_out_layers, kernel_sizes, strides):
    conv1 = conv2d(inputs=x, num_outputs=num_out_layers[0], kernel_size=kernel_sizes[0], activation_fn=tf.nn.relu, scope='conv1')
    conv2 = conv2d(inputs=conv1, num_outputs=num_out_layers[1], kernel_size=kernel_sizes[1], activation_fn=tf.nn.relu, scope='conv2')
    return conv2

def feature_extractor(image_tf):
    feature = []
    #image_tf = tf.expand_dims(image_tf, [3])
    with tf.variable_scope('conv_block1'): # H
      conv1 = _conv_block(image_tf, ([64, 64]), (3, 3), (1, 1))
      feature.append(conv1)
      maxpool1 = slim.max_pool2d(conv1, 2, stride=2, padding = 'SAME')
    with tf.variable_scope('conv_block2'):
      conv2 = _conv_block(maxpool1, ([64, 64]), (3, 3), (1, 1))
      feature.append(conv2)
      maxpool2 = slim.max_pool2d(conv2, 2, stride=2, padding = 'SAME')
    with tf.variable_scope('conv_block3'):
      conv3 = _conv_block(maxpool2, ([128, 128]), (3, 3), (1, 1))
      feature.append(conv3)
      maxpool3 = slim.max_pool2d(conv3, 2, stride=2, padding = 'SAME')
    with tf.variable_scope('conv_block4'):
      conv4 = _conv_block(maxpool3, ([128, 128]), (3, 3), (1, 1))
      feature.append(conv4)
    
    return feature

def cost_volume(c1, warp, search_range):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: Level of the feature pyramid of Image1
        warp: Warped level of the feature pyramid of image22
        search_range: Search range (maximum displacement)
    """
    padded_lvl = tf.pad(warp, [[0, 0], [search_range, search_range], [search_range, search_range], [0, 0]])
    _, h, w, _ = tf.unstack(tf.shape(c1))
    max_offset = search_range * 2 + 1

    cost_vol = []
    for y in range(0, max_offset):
        for x in range(0, max_offset):
            slice = tf.slice(padded_lvl, [0, y, x, 0], [-1, h, w, -1])
            cost = tf.reduce_mean(c1 * slice, axis=3, keepdims=True)
            cost_vol.append(cost)
    cost_vol = tf.concat(cost_vol, axis=3)
    cost_vol = tf.nn.leaky_relu(cost_vol, alpha=0.1)

    return cost_vol


def _H_model(input1, input2, is_training):
    batch_size = tf.shape(input1)[0]
  
    with tf.variable_scope('feature_extract', reuse = None): 
      feature1 = feature_extractor(input1)
    with tf.variable_scope('feature_extract', reuse = True): # H
      feature2 = feature_extractor(input2)
      
    # Dropout parameter
    keep_prob = 0.5 if is_training==True else 1.0
    
    
    # Regression Net1
    with tf.variable_scope('Reggression_Net1'):    
      search_range = 16
      global_correlation = cost_volume(tf.nn.l2_normalize(feature1[-1],axis=3), tf.nn.l2_normalize(feature2[-1],axis=3), search_range)   
      #3-convolution layers
      net1_conv1 = conv2d(inputs=global_correlation, num_outputs=512, kernel_size=3, activation_fn=tf.nn.relu)
      net1_conv2 = conv2d(inputs=net1_conv1, num_outputs=512, kernel_size=3, activation_fn=tf.nn.relu)
      net1_conv3 = conv2d(inputs=net1_conv2, num_outputs=512, kernel_size=3, activation_fn=tf.nn.relu)  
      # Flatten dropout_conv4
      net1_flat = slim.flatten(net1_conv3)
      # Two fully-connected layers
      with tf.variable_scope('net1_fc1'):
        net1_fc1 = slim.fully_connected(net1_flat, 1024, activation_fn=tf.nn.relu)
        net1_fc1 = slim.dropout(net1_fc1, keep_prob)
      with tf.variable_scope('net1_fc2'):
        net1_fc2 = slim.fully_connected(net1_fc1, 8, activation_fn=None) #BATCH_SIZE x 8
    
    net1_f = tf.expand_dims(net1_fc2, [2])
    patch_size = 32.
    H1 = solve_DLT(net1_f/4., patch_size)
    M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
                  [0., patch_size / 2.0, patch_size / 2.0],
                  [0., 0., 1.]]).astype(np.float32)
    M_tensor = tf.constant(M, tf.float32)
    M_tile = tf.tile(tf.expand_dims(M_tensor, [0]), [batch_size, 1, 1])
    M_inv = np.linalg.inv(M)
    M_tensor_inv = tf.constant(M_inv, tf.float32)
    M_tile_inv = tf.tile(tf.expand_dims(M_tensor_inv, [0]), [batch_size, 1, 1])
    H1 = tf.matmul(tf.matmul(M_tile_inv, H1), M_tile)
    feature2_warp = transform(tf.nn.l2_normalize(feature2[-2],axis=3), H1)
    
    
    # Regression Net2
    with tf.variable_scope('Reggression_Net2'):    
      search_range = 8
      local_correlation_2 = cost_volume(tf.nn.l2_normalize(feature1[-2],axis=3), feature2_warp, search_range)   
      #3-convolution layers
      net2_conv1 = conv2d(inputs=local_correlation_2, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu)
      net2_conv2 = conv2d(inputs=net2_conv1, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu)
      net2_conv3 = conv2d(inputs=net2_conv2, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu, stride=2)  
      # Flatten dropout_conv4
      net2_flat = slim.flatten(net2_conv3)
      # Two fully-connected layers
      with tf.variable_scope('net2_fc1'):
        net2_fc1 = slim.fully_connected(net2_flat, 512, activation_fn=tf.nn.relu)
        net2_fc1 = slim.dropout(net2_fc1, keep_prob)
      with tf.variable_scope('net2_fc2'):
        net2_fc2 = slim.fully_connected(net2_fc1, 8, activation_fn=None) #BATCH_SIZE x 8
    
    net2_f = tf.expand_dims(net2_fc2, [2])
    patch_size = 64.
    H2 = solve_DLT((net1_f+net2_f)/2., patch_size)
    M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
                  [0., patch_size / 2.0, patch_size / 2.0],
                  [0., 0., 1.]]).astype(np.float32)
    M_tensor = tf.constant(M, tf.float32)
    M_tile = tf.tile(tf.expand_dims(M_tensor, [0]), [batch_size, 1, 1])
    M_inv = np.linalg.inv(M)
    M_tensor_inv = tf.constant(M_inv, tf.float32)
    M_tile_inv = tf.tile(tf.expand_dims(M_tensor_inv, [0]), [batch_size, 1, 1])
    H2 = tf.matmul(tf.matmul(M_tile_inv, H2), M_tile)
    feature3_warp = transform(tf.nn.l2_normalize(feature2[-3],axis=3), H2)
    
    
    # Regression Net3
    with tf.variable_scope('Reggression_Net3'):    
      search_range = 4
      local_correlation_3 = cost_volume(tf.nn.l2_normalize(feature1[-3],axis=3), feature3_warp, search_range)   
      #3-convolution layers
      net3_conv1 = conv2d(inputs=local_correlation_3, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
      net3_conv2 = conv2d(inputs=net3_conv1, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu, stride=2)
      net3_conv3 = conv2d(inputs=net3_conv2, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu, stride=2)  
      # Flatten dropout_conv4
      net3_flat = slim.flatten(net3_conv3)
      # Two fully-connected layers
      with tf.variable_scope('net3_fc1'):
        net3_fc1 = slim.fully_connected(net3_flat, 256, activation_fn=tf.nn.relu)
        net3_fc1 = slim.dropout(net3_fc1, keep_prob)
      with tf.variable_scope('net3_fc2'):
        net3_fc2 = slim.fully_connected(net3_fc1, 8, activation_fn=None) #BATCH_SIZE x 8
      
    net3_f = tf.expand_dims(net3_fc2, [2])
      
    
    return net1_f, net2_f, net3_f