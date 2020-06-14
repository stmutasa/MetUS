""" Evaluating the localization network on a CPU or GPU """

import os
import time

import Model as network
import tensorflow as tf
import SODTester as SDT
import SODLoader as SDL
import SOD_Display as SDD
import glob
from pathlib import Path
import numpy as np

sdl= SDL.SODLoader('/home/stmutasa/Code/Datasets/Scaphoid/')
sdd = SDD.SOD_Display()

_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the data variables
tf.app.flags.DEFINE_string('data_dir', 'data/test/', """Path to the data directory.""")
tf.app.flags.DEFINE_string('training_dir', 'training/', """Path to the training directory.""")
tf.app.flags.DEFINE_integer('box_dims', 384, """dimensions to save files""")
tf.app.flags.DEFINE_integer('network_dims', 64, """dimensions of the network input""")
tf.app.flags.DEFINE_integer('epoch_size', 180, """How many examples""")
tf.app.flags.DEFINE_integer('batch_size', 180, """Number of images to process in a batch.""")

# Hyperparameters:
tf.app.flags.DEFINE_float('dropout_factor', 0.5, """ Keep probability""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.999, """ The decay rate for the moving average tracker""")

# Directory control
tf.app.flags.DEFINE_string('train_dir', 'training/', """Directory to write event logs and save checkpoint files""")
tf.app.flags.DEFINE_string('RunInfo', 'Center/', """Unique file name for this training run""")
tf.app.flags.DEFINE_integer('GPU', 0, """Which GPU to use""")

# Define a custom training class
def test():


    # Makes this the default graph where all ops will be added
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        # Define phase of training
        phase_train = tf.placeholder(tf.bool)

        # Load the images and labels.
        iterator = network.inputs(training=False, skip=True)
        data = iterator.get_next()

        # Define input shape
        data['data'] = tf.reshape(data['data'], [FLAGS.batch_size, FLAGS.network_dims, FLAGS.network_dims])

        # Perform the forward pass:
        logits = network.forward_pass(data['data'], phase_train=phase_train)
        softmax = tf.nn.softmax(logits)

        # Labels
        labels = data['label']

        # Initialize variables operation
        var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Restore moving average of the variables
        var_ema = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay)

        # Define variables to restore
        var_restore = var_ema.variables_to_restore()

        # Initialize the saver
        saver = tf.train.Saver(var_restore, max_to_keep=3)

        # Trackers for best performers
        best_AUC, best_epoch = 0.1, 0

        while True:

            # Allow memory placement growth
            config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as mon_sess:

                # Print run info
                print("*** Validation Run %s on GPU %s ****" % (FLAGS.RunInfo, FLAGS.GPU))

                # Retreive the checkpoint
                ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir+FLAGS.RunInfo)

                # Initialize the variables
                mon_sess.run(var_init)
                mon_sess.run(iterator.initializer)

                if ckpt and ckpt.model_checkpoint_path:

                    # Restore the model
                    saver.restore(mon_sess, ckpt.model_checkpoint_path)

                    # Extract the epoch
                    Epoch = ckpt.model_checkpoint_path.split('/')[-1].split('Epoch')[-1]

                else:
                    print ('No checkpoint file found')
                    break

                # Initialize the step counter
                step = 0

                # Set the max step count
                max_steps = int(FLAGS.epoch_size / FLAGS.batch_size)

                # Tester instance
                sdt = SDT.SODTester(True, False)
                sdt.AUC = 0.1

                try:
                    while step < max_steps:

                        # Load some metrics for testing
                        _lbls, _logits, _ptID = mon_sess.run([labels, logits, data['accno']], feed_dict={phase_train: False})

                        # Combine predictions
                        _data, lbl, logitz = sdt.combine_predictions(_lbls, _logits, _ptID, FLAGS.batch_size)

                        # Increment step
                        step += 1

                except tf.errors.OutOfRangeError:
                    print('Done with Training - Epoch limit reached')

                finally:

                    # Calculate final MAE and ACC
                    sdt.calculate_metrics(logitz, lbl, 1, step)
                    sdt.retreive_metrics_classification(Epoch, True)
                    print('------ Current Best AUC: %.4f (Epoch: %s) --------' % (best_AUC, best_epoch))

                    # Lets save runs below 0.8
                    if sdt.AUC >= best_AUC:
                        # Save the checkpoint
                        print(" ---------------- SAVING THIS ONE %s", ckpt.model_checkpoint_path)

                        # Define the filenames
                        checkpoint_file = os.path.join('testing/' + FLAGS.RunInfo, ('Epoch_%s_AUC_%0.3f' % (Epoch, sdt.AUC)))
                        csv_file = os.path.join('testing/' + FLAGS.RunInfo, ('%s_E_%s_AUC_%0.2f.csv' % (FLAGS.RunInfo[:-1], Epoch, sdt.AUC)))

                        # Save the checkpoint
                        saver.save(mon_sess, checkpoint_file)
                        sdl.save_Dict_CSV(_data, csv_file)

                        # Save a new best MAE
                        best_AUC = sdt.AUC
                        best_epoch = Epoch

                    # Shut down the session
                    mon_sess.close()

            # Break if this is the final checkpoint
            if '3000' in Epoch: break

            # Print divider
            print('-' * 70)

            # Otherwise check folder for changes
            filecheck = glob.glob(FLAGS.train_dir+FLAGS.RunInfo + '*')
            newfilec = filecheck

            # Sleep if no changes
            while filecheck == newfilec:

                # Sleep an amount of time proportional to the epoch size
                time.sleep(int(FLAGS.epoch_size * 0.05))

                # Recheck the folder for changes
                newfilec = glob.glob(FLAGS.train_dir+FLAGS.RunInfo + '*')



def main(argv=None):  # pylint: disable=unused-argument
    time.sleep(60)
    if tf.gfile.Exists('testing/'):
        tf.gfile.DeleteRecursively('testing/')
    tf.gfile.MakeDirs('testing/')
    test()


if __name__ == '__main__':
    tf.app.run()