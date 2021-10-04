import tensorflow as tf
import numpy as np
from collections import OrderedDict
import os
import glob
import cv2


rng = np.random.RandomState(2020)

class DataLoader(object):
    def __init__(self, image_folder):
        self.dir = image_folder
        self.images = OrderedDict()
        self.setup()

    
    def __call__(self, batch_size):
        image_info_list = list(self.images.values())
        length = image_info_list[0]['length']

        def image_clip_generator():
            while True:
                image_clip = []
                frame_id = rng.randint(0, length-1)
                image_clip.append(np_load_input(image_info_list[2]['frame'][frame_id]))
                image_clip.append(np_load_input(image_info_list[3]['frame'][frame_id]))
                image_clip.append(np_load_input(image_info_list[0]['frame'][frame_id]))
                image_clip.append(np_load_input(image_info_list[1]['frame'][frame_id]))
                image_clip = np.concatenate(image_clip, axis=2)
                yield image_clip

        dataset = tf.data.Dataset.from_generator(generator=image_clip_generator, output_types=tf.float32, output_shapes=[None, None, 12])
        print('generator dataset, {}'.format(dataset))
        dataset = dataset.prefetch(buffer_size=32)
        dataset = dataset.shuffle(buffer_size=32).batch(batch_size)
        print('epoch dataset, {}'.format(dataset))

        return dataset

    def __getitem__(self, image_name):
        assert image_name in self.images.keys(), 'image = {} is not in {}!'.format(image_name, self.images.keys())
        return self.images[image_name]

    def setup(self):
        images = glob.glob(os.path.join(self.dir, '*'))
        for image in sorted(images):
            image_name = image.split('/')[-1]
            if image_name == 'warp1' or image_name == 'warp2' or image_name == 'mask1' or image_name == 'mask2':
                self.images[image_name] = {}
                self.images[image_name]['path'] = image
                self.images[image_name]['frame'] = glob.glob(os.path.join(image, '*.jpg'))
                self.images[image_name]['frame'].sort()
                self.images[image_name]['length'] = len(self.images[image_name]['frame'])

        print(self.images.keys())

    def get_image_clips(self, index):
        batch = []
        image_info_list = list(self.images.values())
        
        batch.append(np_load_input(image_info_list[2]['frame'][index]))
        batch.append(np_load_input(image_info_list[3]['frame'][index]))
       
        return np.concatenate(batch, axis=2)



def np_load_input(filename):
    image_decoded = cv2.imread(filename)
    height = image_decoded.shape[0] 
    width = image_decoded.shape[1]  
    
    
    # define the max image size to avoid the OOM error
    if height > 1024:
        height = 1024
    if width > 1024:
        width = 1024

    image_resized = cv2.resize(image_decoded, (width, height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = image_resized / 127.5 - 1.
    return image_resized



def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')




