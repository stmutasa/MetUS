"""
Pre processes all of the lymph node inputs and saves as tfrecords
Also handles loading tfrecords
"""

import numpy as np
import tensorflow as tf
import SODLoader as SDL
import SOD_Display as SDD
from pathlib import Path
import os
from random import shuffle
import logging

# Setup logger 1:CRITICAL 2: ERROR 3 WARNING 4 INFO 5 DEBUG
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh = logging.FileHandler('Logfile.txt')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

# Define the flags class for variables
FLAGS = tf.app.flags.FLAGS

# Define the data directories to use
home_dir = '/home/stmutasa/PycharmProjects/Datasets/BreastData/US/LN2/'
test_data = home_dir + 'Testing/'
train_data = home_dir + 'Training/'

sdl = SDL.SODLoader(data_root=home_dir)
sdd = SDD.SOD_Display()


def pre_proc_train(box_dims=384):

    """
    Pre processes the input for the training data
    :param box_dims: dimensions of the saved images
    :return:
    """

    # Load the files and randomly shuffle them
    filenames = sdl.retreive_filelist('nii.gz', True, train_data)
    filenames = [x for x in filenames if 'label' in x]
    shuffle(filenames)

    # Global variables
    display, counter, data, data_test, index, pt = [], [0, 0], {}, {}, 0, 0

    for file in filenames:

        # Load the label volume first
        try:
            segments = sdl.load_NIFTY(file)
        except Exception as e:
            logger.warning('Segment Load Error: %s' %e)
            continue

        # Retreive the patient information
        try:
            fbase, fdir = os.path.basename(file), os.path.dirname(file)
            accno = file.split('/')[-2]
            proj = fbase.replace(' ', '').replace('AXILLA', '').replace(accno, '').replace('X', '').replace('-label.nii.gz', '')
            if 'No' in fdir.split('/')[-2]: label = 0
            else: label = 1
        except Exception as e:
            logger.warning('View/Accno error: %s' % e)
            continue

        # Set destination filename info
        view = accno + '_' + proj

        # Now load the image itsself
        try:
            volfile = fdir + '/' + fbase[:2].replace('X', '') + fbase[2:].replace('-label', '')
            image = sdl.load_NIFTY(volfile)
        except Exception as e:
            logger.warning('Image Load Error: %s' %e)
            continue

        # Normalize here since we want it normed to the general surrounding fat
        image = sdl.adaptive_normalization(image)

        # Apply the segmentation to the image
        image *= segments.astype(np.uint8)

        """
            The nodes get as big as 437 pixels across
            Make a regular crop that size, then another that's normalized to the individual node
            Save both as two channels
        """
        crop_size = 440
        blob, cn = sdl.largest_blob(segments)
        radius = int(np.sum(blob) ** (1 / 3) * 2.5) * 3
        crop_image, _ = sdl.generate_box(image, cn, crop_size, dim3d=False)
        norm_image, _ = sdl.generate_box(image, cn, radius, dim3d=False)

        # Now resize and combine the images into one
        BD = box_dims
        crop_image, norm_image = sdl.zoom_2D(crop_image, [BD, BD]), sdl.zoom_2D(norm_image, [BD, BD])
        crop_image, norm_image = np.expand_dims(crop_image, -1), np.expand_dims(norm_image, -1)
        final_image = np.concatenate([crop_image, norm_image], -1).astype(np.float32)

        # Save the data
        data[index] = {'data': final_image.astype(np.float16), 'label': label, 'group': 'train', 'view': view, 'accno': accno}

        # Increment counters
        index += 1
        pt += 1
        del image

    # Save the data.
    sdl.save_dict_filetypes(data[0])
    sdl.save_segregated_tfrecords(2, data, 'accno', 'data/train/LNs')


# Load the protobuf
def load_data(training=True):

    """
    Loads the protocol buffer into a form to send to shuffle. To oversample classes we made some mods...
    Load with parallel interleave -> Prefetch -> Large Shuffle -> Parse labels -> Undersample map -> Flat Map
    -> Prefetch -> Oversample Map -> Flat Map -> Small shuffle -> Prefetch -> Parse images -> Augment -> Prefetch -> Batch
    """

    # Lambda functions for retreiving our protobuf
    _parse_all = lambda dataset: sdl.load_tfrecords(dataset, [FLAGS.box_dims, FLAGS.box_dims, 2], tf.float16)

    # Load tfrecords with parallel interleave if training
    if training:
        filenames = sdl.retreive_filelist('tfrecords', False, path=FLAGS.data_dir)
        files = tf.data.Dataset.list_files(os.path.join(FLAGS.data_dir, '*.tfrecords'))
        dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=len(filenames),
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
        print('******** Loading Files: ', filenames)
    else:
        files = sdl.retreive_filelist('tfrecords', False, path=FLAGS.data_dir)
        dataset = tf.data.TFRecordDataset(files, num_parallel_reads=1)
        print('******** Loading Files: ', files)

    # Shuffle and repeat if training phase
    if training:

        # Define our undersample and oversample filtering functions
        _filter_fn = lambda x: sdl.undersample_filter(x['label'], actual_dists=[0.62, 0.38], desired_dists=[.5, .5])
        _undersample_filter = lambda x: dataset.filter(_filter_fn)
        _oversample_filter = lambda x: tf.data.Dataset.from_tensors(x).repeat(
            sdl.oversample_class(x['label'], actual_dists=[0.62, 0.38], desired_dists=[.5, .5]))

        # Large shuffle, repeat for 100 epochs then parse the labels only
        dataset = dataset.shuffle(buffer_size=FLAGS.epoch_size)
        dataset = dataset.repeat(FLAGS.repeats)
        dataset = dataset.map(_parse_all, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Now we have the labels, undersample then oversample.
        # Map allows us to do it in parallel and flat_map's identity function merges the survivors
        dataset = dataset.map(_undersample_filter, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.flat_map(lambda x: x)
        dataset = dataset.map(_oversample_filter, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.flat_map(lambda x: x)

        # Now perform a small shuffle in case we duplicated neighbors, then prefetch before the final map
        dataset = dataset.shuffle(buffer_size=20)

    else:
        dataset = dataset.map(_parse_all, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    scope = 'data_augmentation' if training else 'input'
    with tf.name_scope(scope):
        dataset = dataset.map(DataPreprocessor(training), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batch and prefetch
    if training:
        dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.batch(FLAGS.batch_size)

    # Make an initializable iterator
    iterator = dataset.make_initializable_iterator()

    # Return data as a dictionary
    return iterator


class DataPreprocessor(object):

    # Applies transformations to dataset

  def __init__(self, distords):
    self._distords = distords

  def __call__(self, record):

    """Process img for training or eval."""
    image = record['data']

    if self._distords:  # Training

        # Data Augmentation ------------------ Contrast, brightness, noise, rotate, shear, crop, flip

        # Select random slice to use, use norm distribution around center. Remember z = 16
        slice = tf.squeeze(tf.random.uniform([1], 0, 2, tf.int32))
        image = tf.squeeze(image[:,:, slice])
        #image = tf.cond(slice > 0, lambda: tf.squeeze(image[:,:, 1]), lambda: tf.squeeze(image[:,:, 0]))

        # Now augument this slice. First calc rotation parameters
        angle = tf.squeeze(tf.random.uniform([1], -0.78, 0.78, tf.float32))

        # Random rotate
        image = tf.contrib.image.rotate(image, angle, interpolation='BILINEAR')

        # Then randomly flip
        image = tf.expand_dims(image, -1)
        image = tf.image.random_flip_left_right(tf.image.random_flip_up_down(image))

        # Random brightness/contrast
        image = tf.image.random_contrast(image, lower=0.95, upper=1.05)

        # Random center crop
        ninety = int(FLAGS.box_dims * 0.9)
        image = tf.image.random_crop(image, (ninety, ninety, 1))

        # Reshape image
        image = tf.image.resize_images(image, [FLAGS.network_dims, FLAGS.network_dims])

        # For noise, first randomly determine how 'noisy' this study will be
        T_noise = tf.random_uniform([1], 0, 0.1)

        # Create a poisson noise array
        noise = tf.random_uniform(shape=[FLAGS.network_dims, FLAGS.network_dims, 1], minval=-T_noise, maxval=T_noise)

        # Add the poisson noise
        image = tf.add(image, tf.cast(noise, tf.float32))

    else: # Validation

        # Resize to network size
        image = tf.expand_dims(image, -1)
        image = tf.image.resize_images(image, [FLAGS.network_dims, FLAGS.network_dims], tf.compat.v1.image.ResizeMethod.BICUBIC)
        record['img_small'] = tf.image.resize_images(image, [FLAGS.network_dims//8, FLAGS.network_dims//8], tf.compat.v1.image.ResizeMethod.BICUBIC)

    # Make record image
    record['data'] = image

    return record

#pre_proc_train(512)