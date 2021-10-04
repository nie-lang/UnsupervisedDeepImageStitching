import tensorflow as tf
import cv2
import numpy as np
from PIL import Image


def Stitching_Domain_STN(inputs, H_tf, size, resized_shift):
    """Spatial Transformer Layer

    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and edited by David Dao for Tensorflow.

    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    theta: float
        The output of the
        localisation network should be [num_batch, 6].
    out_size: tuple of two ints
        The size of the output of the network (height, width)
    """

    def _repeat(x, n_repeats):
        # Process
        # dim2 = width
        # dim1 = width*height
        # v = tf.range(num_batch)*dim1
        # print 'old v:', v # num_batch
        # print 'new v:', tf.reshape(v, (-1, 1)) # widthx1
        # n_repeats = 20
        # rep = tf.transpose(tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0]) # 1 x out_width*out_height
        # print rep
        # rep = tf.cast(rep, 'int32')
        # v = tf.matmul(tf.reshape(v, (-1, 1)), rep) # v: num_batch x (out_width*out_height)
        # print '--final v:\n', v.eval()
        # # v is the base. For parallel computing.
        # with tf.variable_scope('_repeat'):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, 'int32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        # with tf.variable_scope('_interpolate'):
        # constants
        num_batch = tf.shape(im)[0]
        height = tf.shape(im)[1]
        width = tf.shape(im)[2]
        channels = tf.shape(im)[3]

        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')
        height_f = tf.cast(height, 'float32')
        width_f = tf.cast(width, 'float32')
        out_height = out_size[0]
        out_width = out_size[1]
        zero = tf.zeros([], dtype='int32')
        max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
        max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

        # scale indices from [-1, 1] to [0, width/height]
        # x = (x + 1.0) * (width_f) / 2.0
        # y = (y + 1.0) * (height_f) / 2.0

        # do sampling
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x)  
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        dim2 = width
        dim1 = width * height
        base = _repeat(tf.range(num_batch) * dim1, out_height * out_width)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        #print(im.shape)
        im_flat = tf.reshape(im, tf.stack([-1, channels]))
        im_flat = tf.cast(im_flat, 'float32')
        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        # and finally calculate interpolated values
        x0_f = tf.cast(x0, 'float32')
        x1_f = tf.cast(x1, 'float32')
        y0_f = tf.cast(y0, 'float32')
        y1_f = tf.cast(y1, 'float32')
        wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
        wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
        wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
        wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
        output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
        return output
        # return Ia

    def _meshgrid(width_max, width_min, height_max, height_min):
        # with tf.variable_scope('_meshgrid'):
        #shift = (304. - 128.) / 2.
        
        # shift_h = tf.cast((height - h)/2, tf.float32)
        # shift_w = tf.cast((width - w)/2, tf.float32)
        # height = tf.cast(height, tf.float32)
        # width = tf.cast(width, tf.float32)

        width = width_max - width_min
        height = height_max - height_min
        tf.linspace(width_min,  width_max, tf.cast(width, tf.int32))
        tf.ones(shape=tf.stack([tf.cast(height, tf.int32), 1]))
        x_t = tf.matmul(tf.ones(shape=tf.stack([tf.cast(height, tf.int32), 1])),
                        tf.transpose(tf.expand_dims(tf.linspace(width_min,  width_max, tf.cast(width, tf.int32)), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.linspace(height_min,  height_max, tf.cast(height, tf.int32)), 1),
                        tf.ones(shape=tf.stack([1, tf.cast(width, tf.int32)])))
        

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        grid = tf.concat([x_t_flat, y_t_flat, ones], 0)
        # sess = tf.get_default_session()
        # print '--grid: \n', grid.eval() # (session=sess.as_default())
        return grid

    def _transform(image_tf, H_tf, width_max, width_min, height_max, height_min):
        # with tf.variable_scope('_transform'):
        num_batch = tf.shape(image_tf)[0]
        num_height = tf.shape(image_tf)[1]
        num_width = tf.shape(image_tf)[2]
        num_channels = tf.shape(image_tf)[3]
        #  Changed
        # theta = tf.reshape(theta, (-1, 2, 3))
        H_tf = tf.reshape(H_tf, (-1, 3, 3))
        H_tf = tf.cast(H_tf, 'float32')

        #  Added: add two matrices M and B defined as follows in
        # order to perform the equation: H x M x [xs...;ys...;1s...] + H x [width/2...;height/2...;0...]
        H_tf_shape = H_tf.get_shape().as_list()
        # initial

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        # out_height = num_height + 2* (num_height//2) + 2* (num_height//5)
        # out_width = num_width + 2* (num_width//2) + 2* (num_width//5)
        out_width = tf.cast(width_max - width_min, tf.int32)
        out_height = tf.cast(height_max - height_min, tf.int32)
        grid = _meshgrid(width_max, width_min, height_max, height_min)
        grid = tf.expand_dims(grid, 0)
        grid = tf.reshape(grid, [-1])
        grid = tf.tile(grid, tf.stack([num_batch]))  # stack num_batch grids
        grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))

        # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
        T_g = tf.matmul(H_tf, grid)
        # T_g = grid
        x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
        # Ty changed
        # y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
        y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
        # Ty added
        t_s = tf.slice(T_g, [0, 2, 0], [-1, 1, -1])
        # The problem may be here as a general homo does not preserve the parallelism
        # while an affine transformation preserves it.
        t_s_flat = tf.reshape(t_s, [-1])

        # # Avoid zero division
        # zero = tf.constant(0, dtype=tf.float32)
        one = tf.constant(1, dtype=tf.float32)

        # smaller
        small = tf.constant(1e-7, dtype=tf.float32)
        smallers = 1e-6 * (one - tf.cast(tf.greater_equal(tf.abs(t_s_flat), small), tf.float32))

        t_s_flat = t_s_flat + smallers


        x_s_flat = tf.reshape(x_s, [-1]) / t_s_flat
        y_s_flat = tf.reshape(y_s, [-1]) / t_s_flat

        input_transformed = _interpolate(image_tf, x_s_flat, y_s_flat, (out_height, out_width))

        output = tf.reshape(input_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))
        return output

    ################################################
    ################################################
    ################################################
    pts_1_tile = tf.tile(size, [1, 4, 1])
    tmp = tf.expand_dims(tf.constant([0., 0., 1., 0., 0., 1., 1., 1.], shape=(8,1), dtype = tf.float32), [0])
    pts_1 = pts_1_tile*tmp
    pts_2 = tf.add(resized_shift, pts_1)
	
    print("pts_1 shape")
    print(pts_1.shape)
    pts1_list = tf.split(pts_1, 8, axis = 1)
    print(len(pts1_list))
    print(pts1_list[0].shape)
    pts2_list = tf.split(pts_2, 8, axis = 1)
    #pts_list = tf.concat([pts1_list, pts2_list], axis = 1)
    pts_list = pts1_list + pts2_list
    print(len(pts_list))
    print(pts_list[0].shape)
    width_list = [pts_list[i] for i in range(0, 16, 2)]
    height_list = [pts_list[i] for i in range(1, 16, 2)]
    width_list_tf = tf.concat(width_list, axis=1)
    height_list_tf = tf.concat(height_list, axis=1)
    width_max = tf.reduce_max(width_list_tf)
    width_min = tf.reduce_min(width_list_tf)
    height_max = tf.reduce_max(height_list_tf)
    height_min = tf.reduce_min(height_list_tf)
    print("height_min")
    print(height_min.shape)
    out_width = width_max - width_min
    out_height = height_max - height_min

  
    batch_size = tf.shape(inputs)[0]
    #step 1. 
    H_one = tf.eye(3)
    H_one = tf.tile(tf.expand_dims(H_one, [0]), [batch_size, 1, 1])
    img1_tf = inputs[... , 0:3]+1.
    img1_tf = _transform(img1_tf, H_one, width_max, width_min, height_max, height_min)
    
    #step 2.
    warp_tf = inputs[...,3:6]+1.
    warp_tf = _transform(warp_tf, H_tf, width_max, width_min, height_max, height_min)

    img1_tf = img1_tf - 1.
    warp_tf = warp_tf - 1.

    one = tf.ones_like(inputs[... , 0:3], dtype=tf.float32)
    mask1 = _transform(one, H_one, width_max, width_min, height_max, height_min)
    mask2 = _transform(one, H_tf, width_max, width_min, height_max, height_min)
    
    resized_height = out_height - out_height%8
    resized_width = out_width - out_width%8
    img1_tf = tf.image.resize_images(img1_tf, [resized_height, resized_width], method=0)
    warp_tf = tf.image.resize_images(warp_tf, [resized_height, resized_width], method=0)
    mask1 = tf.image.resize_images(mask1, [resized_height, resized_width], method=0)
    mask2 = tf.image.resize_images(mask2, [resized_height, resized_width], method=0)

    output = tf.concat([img1_tf, warp_tf, mask1, mask2], axis=3)
    
    return output






