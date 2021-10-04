import tensorflow as tf
import numpy as np
from collections import OrderedDict
import os
import glob
import cv2

rng = np.random.RandomState(2017)

def np_load_frame(filename, resize_height, resize_width):
    image_decoded = cv2.imread(filename)
    if resize_height != None and resize_width != None:
        image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    else:
        image_resized = image_decoded
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized

     
        
def np_load_size(filename):
    image_decoded = cv2.imread(filename)
    height = image_decoded.shape[0] 
    width = image_decoded.shape[1]  
    size = np.array([width, height], dtype=np.float32)
    return np.expand_dims(size, 1)

class DataLoader(object):
    def __init__(self, data_folder):
        self.dir = data_folder
        self.datas = OrderedDict()
        self.setup()

    def __call__(self, batch_size):
        data_info_list = list(self.datas.values())
        length = data_info_list[0]['length']

        def data_clip_generator():
            #frame_id = 0
            while True:
                data_clip = []
                size_clip = []
                frame_id = rng.randint(0, length-1)
                #######inputs
                data_clip.append(np_load_frame(data_info_list[0]['frame'][frame_id], 128, 128))
                data_clip.append(np_load_frame(data_info_list[1]['frame'][frame_id], 128, 128))
                data_clip = np.concatenate(data_clip, axis=2)
                #######size
                size_clip.append(np_load_size(data_info_list[0]['frame'][frame_id]))
                size_clip = np.concatenate(size_clip, axis=0)
                
                yield (data_clip, size_clip)

        dataset = tf.data.Dataset.from_generator(generator=data_clip_generator, output_types=(tf.float32, tf.float32),
                                                  output_shapes=([128, 128, 6], [2,1]))
        print('generator dataset, {}'.format(dataset))
        dataset = dataset.prefetch(buffer_size=1000)
        dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)
        print('epoch dataset, {}'.format(dataset))

        return dataset

    def __getitem__(self, data_name):
        assert data_name in self.datas.keys(), 'data = {} is not in {}!'.format(data_name, self.datas.keys())
        return self.datas[data_name]

    def setup(self):
        datas = glob.glob(os.path.join(self.dir, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input1' or data_name == 'input2' :
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['frame'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['frame'].sort()
                self.datas[data_name]['length'] = len(self.datas[data_name]['frame'])
        print(self.datas.keys())
    
    
    # test: get input images
    def get_data_clips(self, index, resize_height, resize_width):
        batch = []
        data_info_list = list(self.datas.values())
        for i in range(0, 2):
            image = np_load_frame(data_info_list[i]['frame'][index], resize_height, resize_width)
            batch.append(image)
        return np.concatenate(batch, axis=2)
    
    # test: get size
    def get_size_clips(self, index):
        batch = []
        data_info_list = list(self.datas.values())
        size = np_load_size(data_info_list[0]['frame'][index])    
        return size




def load(saver, sess, ckpt_path):
    #ckpt_path = 'checkpoints/stitch_rgb__lp_1.0_adv_0.0_gdl_0.0_flow_0.0/model.ckpt-600000'
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')




