import tensorflow as tf
import os

from models import Vgg19_simple_api, reconstruction, seammask_extraction
from loss_functions import intensity_loss
from utils import load, save, DataLoader
import constant
import numpy as np
import tensorlayer as tl


os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU
train_folder = constant.TRAIN_FOLDER
batch_size = constant.TRAIN_BATCH_SIZE
iterations = constant.ITERATIONS
summary_dir = constant.SUMMARY_DIR
snapshot_dir = constant.SNAPSHOT_DIR


# define dataset
with tf.name_scope('dataset'):
    ##########training###############
    train_data_loader = DataLoader(train_folder)
    train_data_dataset = train_data_loader(batch_size=batch_size)
    train_data_it = train_data_dataset.make_one_shot_iterator()
    train_data = train_data_it.get_next()
    train_data.set_shape([batch_size, None, None, 12])
    train_input_tensor = train_data[...,0:6]
    train_mask_tensor = train_data[...,6:12]

    train_inputs = train_input_tensor
    train_mask = train_mask_tensor
        
    print('train inputs = {}'.format(train_inputs))
    



# define training generator function
with tf.variable_scope('Reconstruction', reuse=None):
    print('training = {}'.format(tf.get_variable_scope().name))
    train_lr_stitched, train_hr_stitched = reconstruction(train_inputs)


# set the range of mask1/mask2 to [0,1]
mask1 = (train_mask[...,0:3] + 1.)/2
mask2 = (train_mask[...,3:6] + 1.)/2

# define low-resolution reconstruction loss
lam_lr =  100
if lam_lr != 0:
    # low-resolution content mask
    lr_content_mask1 = tf.image.resize_images(mask1, [256,256],method=0)
    lr_content_mask2 = tf.image.resize_images(mask2, [256,256],method=0)
    # low-resolution input
    lr_input1 = tf.image.resize_images(train_inputs[...,0:3], [256,256],method=0)
    lr_input2 =  tf.image.resize_images(train_inputs[...,3:6], [256,256],method=0)
    
    # low-resolution seam mask
    lr_seam_mask1 = lr_content_mask1*seammask_extraction(lr_content_mask2)
    lr_seam_mask2 = lr_content_mask2*seammask_extraction(lr_content_mask1)
    # low-resolution seam loss
    lr_seam_loss1 = intensity_loss(gen_frames=train_lr_stitched*lr_seam_mask1, gt_frames=lr_input1*lr_seam_mask1, l_num=1)
    lr_seam_loss2 = intensity_loss(gen_frames=train_lr_stitched*lr_seam_mask2, gt_frames=lr_input2*lr_seam_mask2, l_num=1)
    

    train_lr_stitched1_224 = tf.image.resize_images(train_lr_stitched*lr_content_mask1, size=[224, 224], method=0,align_corners=False)  
    train_lr_stitched2_224 = tf.image.resize_images(train_lr_stitched*lr_content_mask2, size=[224, 224], method=0,align_corners=False)  
    train_lr_warp1_224 = tf.image.resize_images(lr_input1*lr_content_mask1, size=[224, 224], method=0, align_corners=False) 
    train_lr_warp2_224 = tf.image.resize_images(lr_input2*lr_content_mask2, size=[224, 224], method=0, align_corners=False)     

    feature_lr_stitched1, _ = Vgg19_simple_api((train_lr_stitched1_224 + 1) / 2, reuse=False)
    feature_lr_stitched2, _  = Vgg19_simple_api((train_lr_stitched2_224 + 1) / 2, reuse=True)
    feature_lr_warp1, _  = Vgg19_simple_api((train_lr_warp1_224 + 1) / 2, reuse = True)
    feature_lr_warp2, _  = Vgg19_simple_api((train_lr_warp2_224 + 1) / 2, reuse = True)
    # low-resolution content loss
    lr_content_loss1 = tl.cost.mean_squared_error(feature_lr_stitched1.outputs, feature_lr_warp1.outputs, is_mean=True)
    lr_content_loss2 = tl.cost.mean_squared_error(feature_lr_stitched2.outputs, feature_lr_warp2.outputs, is_mean=True)
    
    # total low-resolution reconstruction loss
    lr_loss = (lr_seam_loss1 + lr_seam_loss2)*2. + (lr_content_loss1+lr_content_loss2)*1e-6
else:
    lr_loss = tf.constant(0.0, dtype=tf.float32)


# define high-resolution reconstruction loss
lam_hr = 1
if lam_hr != 0:
    # the resolution of high-resolution input
    hr_size = [tf.shape(train_inputs)[1], tf.shape(train_inputs)[2]]
    
    # high-resolution seam mask
    hr_seam_mask1 = tf.image.resize_images(lr_seam_mask1, size=hr_size, method=0, align_corners=False) 
    hr_seam_mask2 = tf.image.resize_images(lr_seam_mask2, size=hr_size, method=0, align_corners=False) 
    # high-resolution seam loss
    hr_seam_loss1 = intensity_loss(gen_frames=train_hr_stitched*hr_seam_mask1, gt_frames=train_inputs[...,0:3]*hr_seam_mask1, l_num=1)
    hr_seam_loss2 = intensity_loss(gen_frames=train_hr_stitched*hr_seam_mask2, gt_frames=train_inputs[...,3:6]*hr_seam_mask2, l_num=1)
    
    # high-resolution content mask
    # mask1 ---- high-resolution content mask 1
    # mask2 ---- high-resolution content mask 2
    
    train_hr_stitched1_224 = tf.image.resize_images(train_hr_stitched*mask1, size=[224, 224], method=0,align_corners=False)  
    train_hr_stitched2_224 = tf.image.resize_images(train_hr_stitched*mask2, size=[224, 224], method=0,align_corners=False)  
    train_hr_warp1_224 = tf.image.resize_images(train_inputs[...,0:3]*mask1, size=[224, 224], method=0, align_corners=False) 
    train_hr_warp2_224 = tf.image.resize_images(train_inputs[...,3:6]*mask2, size=[224, 224], method=0, align_corners=False)     

    _, feature_hr_stitched1 = Vgg19_simple_api((train_hr_stitched1_224 + 1) / 2, reuse=True)
    _, feature_hr_stitched2 = Vgg19_simple_api((train_hr_stitched2_224 + 1) / 2, reuse=True)
    _, feature_hr_warp1 = Vgg19_simple_api((train_hr_warp1_224 + 1) / 2, reuse = True)
    _, feature_hr_warp2 = Vgg19_simple_api((train_hr_warp2_224 + 1) / 2, reuse = True)
    # high-resolution content loss
    hr_content_loss1 = tl.cost.mean_squared_error(feature_hr_stitched1.outputs, feature_hr_warp1.outputs, is_mean=True)
    hr_content_loss2 = tl.cost.mean_squared_error(feature_hr_stitched2.outputs, feature_hr_warp2.outputs, is_mean=True)
    
    # total high-resolution reconstruction loss
    hr_loss = (hr_seam_loss1 + hr_seam_loss2)*2. + (hr_content_loss1+hr_content_loss2)*1e-6
else:
    hr_loss = tf.constant(0.0, dtype=tf.float32)





# define content consistency loss
lam_consistency = 1
if lam_consistency != 0:
    train_hr_stitched_downsample = tf.image.resize_images(train_hr_stitched, size=[256,256], method=0, align_corners=False)  
    consistency_loss = intensity_loss(gen_frames=train_hr_stitched_downsample, gt_frames=train_lr_stitched, l_num=1)
else:
    consistency_loss = tf.constant(0.0, dtype=tf.float32)




with tf.name_scope('training'):
    g_loss = tf.add_n([hr_loss * lam_hr, consistency_loss * lam_consistency, lr_loss * lam_lr], name='g_loss')
    g_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='g_step')
    g_lrate = tf.train.exponential_decay(0.0001, g_step, decay_steps=10000/1, decay_rate=0.98)
    g_optimizer = tf.train.AdamOptimizer(learning_rate=g_lrate, name='g_optimizer')
    g_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='Reconstruction')
    g_train_op = g_optimizer.minimize(g_loss, global_step=g_step, var_list=g_vars, name='g_train_op')




# add all to summaries
tf.summary.scalar(tensor=g_loss, name='g_loss')
tf.summary.scalar(tensor=hr_loss, name='hr_loss')
tf.summary.scalar(tensor=lr_loss, name='lr_loss')
tf.summary.scalar(tensor=consistency_loss, name='consistency_loss')

tf.summary.image(tensor=train_inputs[...,0:3], name='train_input1')
tf.summary.image(tensor=train_inputs[...,3:6], name='train_input2')
tf.summary.image(tensor=mask1, name='mask1')
tf.summary.image(tensor=mask2, name='mask2')
tf.summary.image(tensor=train_hr_stitched, name='train_hr_stitched')
tf.summary.image(tensor=train_lr_stitched, name='train_lr_stitched')
tf.summary.image(tensor=hr_seam_mask1, name='hr_seam_mask1')
tf.summary.image(tensor=hr_seam_mask2, name='hr_seam_mask2')


summary_op = tf.summary.merge_all()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    # summaries
    summary_writer = tf.summary.FileWriter(summary_dir, graph=sess.graph)

    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init successfully!')
    
    
    
    #initialize vgg19
    ###============================= LOAD VGG ===============================###
    print("load vgg19 pretrained model")
    params = []
    if lam_lr != 0:
        vgg19_npy_path = "../vgg19/vgg19.npy"
        #check path
        if not os.path.isfile(vgg19_npy_path):
            print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
            exit()
        else:
            print('checkpoint found')
        #load model
        npz = np.load(vgg19_npy_path, encoding='latin1', allow_pickle=True).item()
        count = 0
        for val in sorted(npz.items()):
            W = np.asarray(val[1][0])
            b = np.asarray(val[1][1])
            print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
            params.extend([W, b])
            count = count + 1
            if count >= 15:
               break
        tl.files.assign_params(sess, params, feature_lr_stitched1)
        print("load vgg19 pretrained model done!")
    

    # tf saver
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)
    restore_var = [v for v in tf.global_variables()]
    loader = tf.train.Saver(var_list=restore_var)
    print(snapshot_dir)
    if os.path.isdir(snapshot_dir):
        ckpt = tf.train.get_checkpoint_state(snapshot_dir)
        if ckpt and ckpt.model_checkpoint_path:
            load(loader, sess, ckpt.model_checkpoint_path)
            print('===========restart from===========')
            print(ckpt.model_checkpoint_path)
        else:
            #print(ckpt.model_checkpoint_path)
            print('No checkpoint file found.')
    else:
        load(loader, sess, snapshot_dir)

    _step, _loss, _summaries = 0, None, None



    print("============starting training===========")
    while _step < iterations:
        try:

            print('Training generator...')
            _, _g_lr, _step, _hr_loss, _consistency_loss, _lr_loss, _g_loss, _summaries = sess.run([g_train_op, g_lrate, g_step, hr_loss, consistency_loss, lr_loss, g_loss, summary_op])
            
            
            if _step % 10 == 0:
                print('GeneratorModel : Step {}, lr = {:.6f}'.format(_step, _g_lr))
                print('                 Global      Loss : ', _g_loss)
                print('                 lr  Loss : ({:.4f} * {:.6f} = {:.4f})'.format(_lr_loss, lam_lr, _lr_loss * lam_lr))
                print('                 hr  Loss : ({:.4f} * {:.4f} = {:.4f})'.format(_hr_loss, lam_hr, _hr_loss * lam_hr))
                print('                 consistency  Loss : ({:.4f} * {:.4f} = {:.4f})'.format( _consistency_loss, lam_consistency, _consistency_loss * lam_consistency))
            if _step % 200 == 0:
                summary_writer.add_summary(_summaries, global_step=_step)
                print('Save summaries...')

            if _step % 100000 == 0:
                save(saver, sess, snapshot_dir, _step)

        except tf.errors.OutOfRangeError:
            print('Finish successfully!')
            save(saver, sess, snapshot_dir, _step)
            break