import tensorflow as tf
import H_model
import output_tensorDLT
import output_tf_spatial_transform



def H_estimator(inputs_aug, inputs, is_training):
    return H_model.H_model(inputs_aug, inputs, is_training)


def output_H_estimator(inputs, size, is_training):
    shift = H_model.H_model_v2(inputs, is_training)
    size_tmp = tf.concat([size,size,size,size],axis=1)/128.
    resized_shift = tf.multiply(shift, size_tmp)
    
    H = output_tensorDLT.solve_SizeDLT(resized_shift, size)  
    
    coarsealignment = output_tf_spatial_transform.Stitching_Domain_STN(inputs, H, size, resized_shift)
    
    return  coarsealignment



def disjoint_augment_image_pair(train_inputs, min_val=-1, max_val=1):
    img1 = train_inputs[...,0:3]
    img2 = train_inputs[...,3:6]
    
    
    # Randomly shift brightness
    random_brightness = tf.random_uniform([], 0.7, 1.3)
    img1_aug = img1 * random_brightness
    random_brightness = tf.random_uniform([], 0.7, 1.3)
    img2_aug = img2 * random_brightness
    
    # Randomly shift color
    random_colors = tf.random_uniform([3], 0.7, 1.3)
    white = tf.ones([tf.shape(img1)[0], tf.shape(img1)[1], tf.shape(img1)[2]])
    color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=3)
    img1_aug  *= color_image

    random_colors = tf.random_uniform([3], 0.7, 1.3)
    color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=3)
    img2_aug  *= color_image
    
    # Saturate
    img1_aug  = tf.clip_by_value(img1_aug,  min_val, max_val)
    img2_aug  = tf.clip_by_value(img2_aug, min_val, max_val)
    
    train_inputs = tf.concat([img1_aug, img2_aug], axis = 3)

    return train_inputs