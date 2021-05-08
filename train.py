# -*- coding: utf-8 -*-
"""
@author: NysanAskar
"""
import logging
logging.getLogger('tensorflow').setLevel(logging.INFO)
logging.getLogger('tensorflow').setLevel(logging.WARNING)

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import math
import datetime
import os

import tensorflow as tf
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#print(tf.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from yolov4 import YoloV4, YoloLoss, anchors_wh
from preprocess import Preprocessor

BATCH_SIZE = 4
TOTAL_CLASSES = 80
TOTAL_EPOCHS = 5
OUTPUT_SHAPE = (416, 416)
TF_RECORDS = './tfrecords_voc'
SHUFFLE_SIZE = 2
#tf.random.set_seed(1)

class Trainer(object):
    def __init__(self,
                 model,
                 initial_epoch,
                 epochs,
                 global_batch_size,
                 initial_learning_rate=0.001):
        self.model = model
        self.initial_epoch = initial_epoch
        self.epochs = epochs
        self.global_batch_size = global_batch_size
        self.loss_objects = [
            YoloLoss(
                num_classes=TOTAL_CLASSES,
                valid_anchors_wh=anchors_wh[0:3]),  # small scale 52x52
            YoloLoss(
                num_classes=TOTAL_CLASSES,
                valid_anchors_wh=anchors_wh[3:6]),  # medium scale 26x26
            YoloLoss(
                num_classes=TOTAL_CLASSES,
                valid_anchors_wh=anchors_wh[6:9]),  # large scale 13x13
        ]
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=initial_learning_rate, epsilon=0.1)
        # for learning rate schedule
        self.current_learning_rate = initial_learning_rate
        self.optimizer.learning_rate = self.current_learning_rate
    @tf.function
    def train_step(self, inputs):
        images, labels = inputs

        with tf.GradientTape() as tape:
            outputs = self.model(images, training=True)
            all_total_losses = []
            all_xy_losses = []
            all_wh_losses = []
            all_class_losses = []
            all_obj_losses = []
            all_class_accuracy = []
            all_mIoU = []
            # iterate over all three scales
            for loss_object, y_pred, y_true in zip(self.loss_objects, outputs,
                                                   labels):
                total_losses, loss_breakdown = loss_object(y_true, y_pred)
                xy_losses, wh_losses, class_losses, obj_losses, class_accuracy_loss, mIoU_losses = loss_breakdown
                total_loss = total_losses *(1.0/BATCH_SIZE)
                xy_loss = xy_losses*(1.0/BATCH_SIZE)
                wh_loss = wh_losses*(1.0/BATCH_SIZE)
                class_loss = class_losses*(1.0/BATCH_SIZE)
                obj_loss = obj_losses*(1.0/BATCH_SIZE)
                class_accuracy = class_accuracy_loss*(1.0/BATCH_SIZE)
                mIoU_loss = mIoU_losses*(1.0/BATCH_SIZE)
                total_loss = tf.reduce_sum(total_loss)
                total_xy_loss = tf.reduce_sum(xy_loss)
                total_wh_loss = tf.reduce_sum(wh_loss)
                total_class_loss = tf.reduce_sum(class_loss)
                total_obj_loss = tf.reduce_sum(obj_loss)
                total_class_accuracy = tf.reduce_sum(class_accuracy)
                total_mIoU = tf.reduce_sum(mIoU_loss)

                all_total_losses.append(total_loss)
                all_xy_losses.append(total_xy_loss)
                all_wh_losses.append(total_wh_loss)
                all_class_losses.append(total_class_loss)
                all_class_accuracy.append(total_class_accuracy)
                all_obj_losses.append(total_obj_loss)
                all_mIoU.append(total_mIoU)

            total_loss_scales = tf.reduce_sum(all_total_losses)
            total_xy_loss_scales = tf.reduce_sum(all_xy_losses)
            total_wh_loss_scales = tf.reduce_sum(all_wh_losses)
            total_class_loss_scales = tf.reduce_sum(all_class_losses)
            total_obj_loss_scales = tf.reduce_sum(all_obj_losses)
            total_class_accuracy_scales = tf.reduce_sum(all_class_accuracy)/3
            total_mIoU_scales = tf.reduce_sum(all_mIoU)/3

        grads = tape.gradient(
            target=total_loss_scales, sources=self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))
        return total_loss_scales, total_class_accuracy_scales, total_mIoU_scales, (total_xy_loss_scales, total_wh_loss_scales, total_class_loss_scales, total_obj_loss_scales)

    @tf.function
    def val_step(self, inputs):
        images, labels = inputs
        outputs = self.model(images, training=True)
        all_total_losses = []
        all_class_accuracy = []
        all_mIoU = []
        for loss_object, y_pred, y_true in zip(self.loss_objects, outputs,
                                                   labels):
            losses, loss_breakdown = loss_object(y_true, y_pred)
            xy_loss, wh_loss, class_loss, obj_loss, class_accuracy_loss, mIoU_losses = loss_breakdown
            loss = losses *(1.0/BATCH_SIZE)
            total_loss = tf.reduce_sum(loss)
            class_accuracy = class_accuracy_loss*(1.0/BATCH_SIZE)
            total_class_accuracy = tf.reduce_sum(class_accuracy)
            mIoU_loss = mIoU_losses*(1.0/BATCH_SIZE)
            total_mIoU = tf.reduce_sum(mIoU_loss)

            all_total_losses.append(total_loss)
            all_class_accuracy.append(total_class_accuracy)
            all_mIoU.append(total_mIoU)

        total_losses_scales = tf.reduce_sum(all_total_losses)
        total_class_accuracy_scales = tf.reduce_sum(all_class_accuracy)/3
        total_mIoU_scales = tf.reduce_sum(all_mIoU)/3

        return total_losses_scales, total_class_accuracy_scales, total_mIoU_scales

    def get_current_time(self):
        return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def run(self, train_dist_dataset, val_dist_dataset):
        current_time = self.get_current_time()
        tf.print('{} Start training...'.format(current_time))

        current_time = self.get_current_time()
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        val_log_dir = 'logs/gradient_tape/' + current_time + '/val'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        for epoch in range(self.initial_epoch, self.epochs + 1):
            tf.print(
                '{} Started epoch {}.'
                .format(self.get_current_time(), epoch))
            total_train_loss = 0.0
            total_train_accuracy = 0.0
            total_train_mIoU = 0.0
            step = 0.0
            num_elements = 0
            for dataset in train_dist_dataset:
                batch_train_loss, batch_train_accuracy, batch_train_mIoU, _ = self.train_step(dataset)
                total_train_loss += batch_train_loss
                total_train_accuracy += batch_train_accuracy
                total_train_mIoU += batch_train_mIoU
                step += 1.0
                for element in dataset:
                    num_elements += 1*2
            total_train_accuracy =  total_train_accuracy/step
            total_train_mIoU =  total_train_mIoU/step

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', total_train_loss, step=epoch)
                tf.summary.scalar('accuracy', total_train_accuracy, step=epoch)
                tf.summary.scalar('mIoU', total_train_mIoU, step=epoch)

            tf.print(
                '{} Epoch {}, train loss {:.1f}, train accuracy {:.1f}%, train mIoU {:.1f}%, {} train images'
                .format(
                    self.get_current_time(), epoch, total_train_loss, total_train_accuracy*100, total_train_mIoU*100, num_elements))

            total_val_loss = 0.0
            total_val_accuracy = 0.0
            total_val_mIoU = 0.0
            step = 0.0
            num_elements = 0
            for dataset in val_dist_dataset:
                batch_val_loss, batch_val_accuracy, batch_val_mIoU = self.val_step(dataset)
                total_val_loss += batch_val_loss
                total_val_accuracy += batch_val_accuracy
                total_val_mIoU += batch_val_mIoU
                step += 1.0
                for element in dataset:
                    num_elements += 1*2
            total_val_accuracy = total_val_accuracy/step
            total_val_mIoU = total_val_mIoU/step

            with val_summary_writer.as_default():
                tf.summary.scalar('loss', total_val_loss, step=epoch)
                tf.summary.scalar('accuracy', total_val_accuracy, step=epoch)
                tf.summary.scalar('mIoU', total_val_mIoU, step=epoch)

            tf.print(
                            '{} Epoch {}, validation loss {:.1f},validation accuracy {:.1f}%, validation mIoU {:.1f}%, {} validation set images'.format(self.get_current_time(), epoch, total_val_loss, total_val_accuracy*100, total_val_mIoU*100, num_elements))

    def save_model(self, epoch, loss):
        # https://github.com/tensorflow/tensorflow/issues/33565
        model_name = './models/model-v1.0.1-epoch-{}-loss-{:.4f}.tf'.format(
            epoch, loss)
        self.model.save_weights(model_name)
        print("Model {} saved.".format(model_name))


def create_dataset(tfrecords, batch_size):
    preprocess = Preprocessor(TOTAL_CLASSES, OUTPUT_SHAPE)

    dataset = tf.data.Dataset.list_files(tfrecords)
    dataset = tf.data.TFRecordDataset(dataset)
    dataset = dataset.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.shuffle(512)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


################################################
def main():

    train_dataset = create_dataset(
        '{}/train*'.format(TF_RECORDS), BATCH_SIZE)
    val_dataset = create_dataset(
        '{}/val*'.format(TF_RECORDS), BATCH_SIZE)
    if not os.path.exists(os.path.join('./models')):
        os.makedirs(os.path.join('./models/'))

    model = YoloV4(
        shape=(416, 416, 3), num_classes=TOTAL_CLASSES)
    model.summary()

    initial_epoch = 1

    trainer = Trainer(
        model=model,
        initial_epoch=initial_epoch,
        epochs=TOTAL_EPOCHS,
        global_batch_size=BATCH_SIZE)
    trainer.run(train_dataset, val_dataset)


if __name__ == '__main__':
    main()
