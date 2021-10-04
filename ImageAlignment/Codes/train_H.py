import tensorflow as tf
import os

from models import H_estimator, disjoint_augment_image_pair
from loss_functions import intensity_loss
from utils import load, save, DataLoader
import constant
import numpy as np


os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU

train_folder = constant.TRAIN_FOLDER
test_folder = constant.TEST_FOLDER

batch_size = constant.TRAIN_BATCH_SIZE
iterations = constant.ITERATIONS

height, width = 128, 128

summary_dir = constant.SUMMARY_DIR
snapshot_dir = constant.SNAPSHOT_DIR


# define dataset
with tf.name_scope('dataset'):
    ##########training###############
    ###input###
    train_data_loader = DataLoader(train_folder)
    train_data_dataset = train_data_loader(batch_size=batch_size)
    train_data_it = train_data_dataset.make_one_shot_iterator()
    (train_input_tensor, train_size_tensor) = train_data_it.get_next()
    train_input_tensor.set_shape([batch_size, height, width, 3*2])
    train_size_tensor.set_shape([batch_size, 2, 1])
    train_inputs = train_input_tensor
    train_size = train_size_tensor
    print('train inputs = {}'.format(train_inputs))
    

#only training dataset augment
with tf.name_scope('disjoint_augment'):
    train_inputs_aug = disjoint_augment_image_pair(train_inputs)



# define training generator function
with tf.variable_scope('generator', reuse=None):
    print('training = {}'.format(tf.get_variable_scope().name))
    train_net1_f, train_net2_f, train_net3_f, train_warp2_H1, train_warp2_H2, train_warp2_H3, train_one_warp_H1, train_one_warp_H2, train_one_warp_H3 = H_estimator(train_inputs_aug, train_inputs, True)
   
   

with tf.name_scope('loss'):
    lam_lp = 1
    loss1 = intensity_loss(gen_frames=train_warp2_H1, gt_frames=train_inputs[...,0:3]*train_one_warp_H1, l_num=1)
    loss2 = intensity_loss(gen_frames=train_warp2_H2, gt_frames=train_inputs[...,0:3]*train_one_warp_H2, l_num=1)
    loss3 = intensity_loss(gen_frames=train_warp2_H3, gt_frames=train_inputs[...,0:3]*train_one_warp_H3, l_num=1)
    lp_loss = 16. * loss1 + 4. * loss2 + 1. * loss3




with tf.name_scope('training'):
    g_loss = tf.add_n([lp_loss * lam_lp], name='g_loss')

    g_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='g_step')
    g_lrate = tf.train.exponential_decay(0.0001, g_step, decay_steps=50000/4, decay_rate=0.96)
    g_optimizer = tf.train.AdamOptimizer(learning_rate=g_lrate, name='g_optimizer')
    g_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    
    grads = g_optimizer.compute_gradients(g_loss, var_list=g_vars)
    for i, (g, v) in enumerate(grads):
      if g is not None:
        grads[i] = (tf.clip_by_norm(g, 3), v)  # clip gradients
    g_train_op = g_optimizer.apply_gradients(grads, global_step=g_step, name='g_train_op')
    

# add all to summaries
tf.summary.scalar(tensor=g_loss, name='g_loss')
tf.summary.scalar(tensor=loss1, name='loss1')
tf.summary.scalar(tensor=loss2, name='loss2')
tf.summary.scalar(tensor=loss3, name='loss3')

tf.summary.image(tensor=train_inputs[...,0:3], name='train_inpu1')
tf.summary.image(tensor=train_inputs[...,3:6], name='train_inpu2')
tf.summary.image(tensor=train_warp2_H3, name='train_warp2_H3')

summary_op = tf.summary.merge_all()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    # summaries
    summary_writer = tf.summary.FileWriter(summary_dir, graph=sess.graph)

    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init successfully!')

    # tf saver
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)
    restore_var = [v for v in tf.global_variables()]
    loader = tf.train.Saver(var_list=restore_var)
    print("snapshot_dir")
    print(snapshot_dir)
    if os.path.isdir(snapshot_dir):
        ckpt = tf.train.get_checkpoint_state(snapshot_dir)
        if ckpt and ckpt.model_checkpoint_path:
            load(loader, sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found.')
    else:
        load(loader, sess, snapshot_dir)

    _step, _loss, _summaries = 0, None, None

    print("============starting training===========")
    while _step < iterations:
        try:
            print('Training generator...')
            _, _g_lr, _step, _lp_loss, _g_loss, _summaries = sess.run([g_train_op, g_lrate, g_step, lp_loss,  g_loss, summary_op])

            if _step % 100 == 0:
                print('GeneratorModel : Step {}, lr = {:.8f}'.format(_step, _g_lr))
                print('                 Global      Loss : ', _g_loss)
                print('                 intensity   Loss : ({:.4f} * {:.4f} = {:.4f})'.format(_lp_loss, lam_lp, _lp_loss * lam_lp))
            if _step % 1000 == 0:
                summary_writer.add_summary(_summaries, global_step=_step)
                print('Save summaries...')

            if _step % 200000 == 0:
                save(saver, sess, snapshot_dir, _step)

        except tf.errors.OutOfRangeError:
            print('Finish successfully!')
            save(saver, sess, snapshot_dir, _step)
            break
