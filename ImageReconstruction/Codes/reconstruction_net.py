import tensorflow as tf
from tensorflow.contrib.layers import conv2d, max_pool2d, conv2d_transpose, fully_connected



    
def resBlock(x):
    conv1 = conv2d(inputs=x, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    conv2 = conv2d(inputs=conv1, num_outputs=64, kernel_size=3, activation_fn=None)
    out = tf.nn.relu(x+conv2)
    return out



def ReconstructionNet(inputs):

    batch_size = tf.shape(inputs)[0]
    height = tf.shape(inputs)[1]
    width = tf.shape(inputs)[2]
    inputs.set_shape([1, None, None, 6])
    
    HR_inputs = inputs
    
    
    ########################################################
    ################### low-resolution branch###############
    ########################################################
    # the input of low-resolution branch
    warp1 = tf.image.resize_images(inputs[...,0:3], [256,256],method=0)
    warp2 = tf.image.resize_images(inputs[...,3:6], [256,256],method=0)
    LR_inputs = tf.concat([warp1, warp2], axis=3)
    #low-resolution reconstruction branch (encoder)
    encoder_conv1_1 = conv2d(inputs=LR_inputs, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    encoder_conv1_2 = conv2d(inputs=encoder_conv1_1, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    encoder_pooling1 = max_pool2d(inputs=encoder_conv1_2, kernel_size=2, padding='SAME')
    encoder_conv2_1 = conv2d(inputs=encoder_pooling1, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    encoder_conv2_2 = conv2d(inputs=encoder_conv2_1, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    encoder_pooling2 = max_pool2d(inputs=encoder_conv2_2, kernel_size=2, padding='SAME')
    encoder_conv3_1 = conv2d(inputs=encoder_pooling2, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu)
    encoder_conv3_2 = conv2d(inputs=encoder_conv3_1, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu)
    encoder_pooling3 = max_pool2d(inputs=encoder_conv3_2, kernel_size=2, padding='SAME')
    encoder_conv4_1 = conv2d(inputs=encoder_pooling3, num_outputs=512, kernel_size=3, activation_fn=tf.nn.relu)
    encoder_conv4_2 = conv2d(inputs=encoder_conv4_1, num_outputs=512, kernel_size=3, activation_fn=tf.nn.relu)
    #low-resolution reconstruction branch (decoder)
    decoder_up1 = conv2d_transpose(inputs=encoder_conv4_2, num_outputs=256, kernel_size=2, stride=2)
    decoder_concat1 = tf.concat([encoder_conv3_2, decoder_up1], axis=3)
    decoder_conv1_1 = conv2d(inputs=decoder_concat1, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu)
    decoder_conv1_2 = conv2d(inputs=decoder_conv1_1, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu)
    decoder_up2 = conv2d_transpose(inputs=decoder_conv1_2, num_outputs=128, kernel_size=2, stride=2)
    decoder_concat2 = tf.concat([encoder_conv2_2, decoder_up2], axis=3)
    decoder_conv2_1 = conv2d(inputs=decoder_concat2, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    decoder_conv2_2 = conv2d(inputs=decoder_conv2_1, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    decoder_up3 = conv2d_transpose(inputs=decoder_conv2_2, num_outputs=64, kernel_size=2, stride=2)
    decoder_concat3 = tf.concat([encoder_conv1_2, decoder_up3], axis=3)
    decoder_conv3_1 = conv2d(inputs=decoder_concat3, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    decoder_conv3_2 = conv2d(inputs=decoder_conv3_1, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    # the output of low-resolution branch
    LR_output = conv2d(inputs=decoder_conv3_2, num_outputs=3, kernel_size=3, activation_fn=None)
    LR_output = tf.tanh(LR_output)
    
    ########################################################
    ################### high-resolution branch###############
    ########################################################
    # the input of high-resolution branch
    LR_SR = tf.image.resize_images(LR_output, [height, width],method=0)
    HR_inputs = tf.concat([HR_inputs, LR_SR], axis=3)   
    # high-resolution reconstruction branch
    HR_conv1 = conv2d(inputs=HR_inputs, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    x = HR_conv1
    for i in range(8):
      x = resBlock(x)
    HR_conv2 = conv2d(inputs=x, num_outputs=64, kernel_size=3, activation_fn=None)
    HR_conv2 = tf.nn.relu(HR_conv1+HR_conv2)
    # the output of high-resolution branch
    HR_output = conv2d(inputs=HR_conv2, num_outputs=3, kernel_size=3, activation_fn=None)
    HR_output = tf.tanh(HR_output)
    
    

    return LR_output, HR_output
