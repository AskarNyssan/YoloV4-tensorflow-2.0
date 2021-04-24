# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 17:52:58 2021

@author: NysanAskar
"""

import numpy as np
import tensorflow as tf
from keras import backend as K
import keras

from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from utils import xywh_to_x1x2y1y2, xywh_to_y1x1y2x2, broadcast_iou, binary_cross_entropy

anchors_wh = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                       [59, 119], [116, 90], [156, 198], [373, 326]],
                      np.float32) / 416

backbone = [
    (32, 3, 1, 'same', 'mish', '0'),
    ["SPP", 1, '1'],
    (64, 3, 1, 'same', 'mish', '2'),
    ["CSP", 2, '3'],
    (128, 1, 1, 'same', 'mish', '4'),
    ["CSP", 8, '5'],
    (256, 1, 1, 'same', 'mish', '6'),
    ["CSP", 8, '7'],
    (512, 1, 1, 'same', 'mish', '8'),
    ["CSP", 4, '9'],
    (1024, 1, 1, 'same', 'mish', '10'),
]

neck = [
    (512, 1, 1, 'same', 'leaky', '0'),# input_3
    (1024, 3, 1, 'same', 'leaky', '1'),
    ["M", '2'],
    (512, 1, 1, 'same', 'leaky', '3'),
    (1024, 3, 1, 'same', 'leaky', '4'),
    (512, 1, 1, 'same', 'leaky', '5'),# output_3
    (256, 1, 1, 'same', 'leaky', '6'),
    "U", #'7'
    ["C",'8'],
    (256, 1, 1, 'same', 'mish', '9'),
    (512, 3, 1, 'same', 'mish', '10'),
    (256, 1, 1, 'same', 'mish', '11'),
    (512, 3, 1, 'same', 'mish', '12'),
    (256, 1, 1, 'same', 'mish', '13'),# output_2
    (128, 1, 1, 'same', 'leaky', '14'),
    "U", #'15'
    ["C",'16'],
    (128, 1, 1, 'same', 'leaky', '17'),
    (256, 3, 1, 'same', 'mish', '18'),
    (128, 1, 1, 'same', 'leaky', '19'),
    (256, 3, 1, 'same', 'mish', '20'),
    (128, 1, 1, 'same', 'leaky', '21'),
    (256, 3, 1, 'same', 'mish', '22'),# output_1
]


head = [
    ["S", 256, '0'],# input_3 -> output_1
    (256, 3, 2, 'valid', 'leaky', '1'),# input_3
    ["C", 256, '2'], # concatanate with input_2
    (512, 3, 1, 'same', 'leaky', '3'),
    (256, 1, 1, 'same', 'leaky', '4'),
    (512, 3, 1, 'same', 'leaky', '5'),
    (256, 1, 1, 'same', 'leaky', '6'),
    ["S", 256, '7'],# output_2
    (512, 3, 2, 'valid', 'leaky', '8'),
    ["C", 512, '9'], # concatanate with input_1
    (1024, 3, 1, 'same', 'leaky', '10'),
    (512, 1, 1, 'same', 'leaky', '11'),# output_2
    (1024, 3, 1, 'same', 'leaky', '12'),
    (512, 1, 1, 'same', 'leaky', '13'),
    ["S", 512, '14'],# output_3
]


class Mish(tf.keras.layers.Layer):
    '''
    Mish Activation Function.
    .. math::
        Mish(x) = x * tanh(softplus(x))
    '''
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

class CNNBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels, bn_act=True, padding='same', activation='leaky',**kwargs):
        super().__init__()
        self.padding = padding
        self.conv = tf.keras.layers.Conv2D(filters = out_channels, use_bias=not bn_act, padding=padding, **kwargs)
        self.bn = tf.keras.layers.BatchNormalization()
        self.leaky = tf.keras.layers.LeakyReLU(0.1)
        self.use_bn_act = bn_act
        self.zero_pad = tf.keras.layers.ZeroPadding2D(padding=(1, 1))
        self.activ_func = activation
        self.mish = Mish()
    def call(self, input_tensor):
        if self.activ_func == 'leaky':
            if self.padding == 'same':
                if self.use_bn_act:
                    return self.leaky(self.bn(self.conv(input_tensor)))
                else:
                    return self.conv(input_tensor)
            else:
                if self.use_bn_act:
                    z = self.zero_pad(input_tensor)
                    return self.leaky(self.bn(self.conv(z)))
                else:
                    z = self.zero_pad(input_tensor)
                    return self.conv(z) # TensorShape([3, 224, 224, 5])
        elif self.activ_func == 'mish':
            if self.padding == 'same':
                if self.use_bn_act:
                    return self.mish(self.bn(self.conv(input_tensor)))
                else:
                    return self.conv(input_tensor)
            else:
                if self.use_bn_act:
                    z = self.zero_pad(input_tensor)
                    return self.mish(self.bn(self.conv(z)))
                else:
                    z = self.zero_pad(input_tensor)
                    return self.conv(z) # TensorShape([3, 224, 224, 5])

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = []
        for _ in range(num_repeats):
            self.layers += [
                keras.Sequential([
                    CNNBlock(channels, kernel_size=1, activation='mish'),
                    CNNBlock(channels, kernel_size=3, padding='same', activation='mish')]
                )]
        self.use_residual = use_residual
        self.num_repeats = num_repeats
    def call(self, input_tensor):
        for layer in self.layers:
            if self.use_residual:
                x = Add()([input_tensor,layer(input_tensor)])
            else:
                x = layer(input_tensor)
        return x # TensorShape([3, 224, 224, 5])

class CSPBlock(tf.keras.layers.Layer):
    def __init__(self, channels, num_res_block=1):
        super().__init__()
        self.conv_1 = CNNBlock(out_channels=channels, padding='valid',
                             activation='mish', kernel_size=3, strides=2)
        self.conv_2 = CNNBlock(out_channels=channels//2, padding='same',
                             activation='mish', kernel_size=1, strides=1)
        self.conv_3 = CNNBlock(out_channels=channels//2, padding='same',
                             activation='mish', kernel_size=1, strides=1)
        self.res_block = ResidualBlock(channels=channels//2, num_repeats=num_res_block)
    def call(self, input_tensor):
        x = self.conv_1(input_tensor)
        x_1 = self.conv_2(x)
        x_2 = self.conv_2(x)
        x_2 = self.res_block(x_2)
        x_2 = self.conv_3(x_2)
        x = tf.concat([x_1, x_2], -1)
        return x # tf.Tensor: shape=(3, 208, 208, 256)

class backbone_layers(tf.keras.layers.Layer):
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()
    def call(self, x):
        outputs = []  # for each scale
        for i, layer in enumerate(self.layers):
            if i in [6, 8, 10]:
                outputs.append(layer(x))
            x = layer(x)
        return outputs
    def _create_conv_layers(self):
        layers = []
        in_channels = self.in_channels
        for module in backbone:
            if isinstance(module, tuple):
                out_channels, kernel_size, strides, padding, ac_faunc, name_layer = module
                layers.append(
                    CNNBlock(
                        out_channels,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding=padding, activation = ac_faunc, name = name_layer))
                in_channels = out_channels
            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(CSPBlock(in_channels*2, num_res_block=num_repeats))
        return layers

class max_pool(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv_1 = CNNBlock(out_channels=512, kernel_size=1,strides=1,
                        padding='same', activation = 'leaky')
        self.max_1 = tf.keras.layers.MaxPool2D((5, 5), strides=1, padding="same")
        self.max_2 = tf.keras.layers.MaxPool2D((9, 9), strides=1, padding="same")
        self.max_3 = tf.keras.layers.MaxPool2D((13, 13), strides=1, padding="same")
    def call(self, x):
        x_1 = self.conv_1(x)
        x_max_1 = self.max_1(x_1)
        x_max_2 = self.max_2(x_1)
        x_max_3 = self.max_3(x_1)
        x = tf.concat([x_1, x_max_1,x_max_2,x_max_3], -1)
        return x


class concat(tf.keras.layers.Layer):
    def __init__(self, channels, ):
        super().__init__()
        self.conc = tf.keras.layers.Concatenate(axis=-1)
        self.conv = CNNBlock(out_channels=channels, kernel_size=1,strides=1,
                        padding='same', activation = 'leaky')
    def call(self, x_1, x_2):
        x_2 = self.conv(x_2)
        x = self.conc([x_1, x_2,])
        return x


class yolov4_neck(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.layers = self._create_conv_layers()
    def call(self, input_tensor):
        outputs = []  # for each scale
        route_connections = []
        inputs = input_tensor[:2]
        x = input_tensor[2]
        for i, layer in enumerate(self.layers):
            if i in [5, 13, 22]:
                outputs.append(layer(x))
            elif isinstance(layer, tf.keras.layers.UpSampling2D):
                route_connections.append(layer(x))
                x = layer(x)
            elif isinstance(layer, concat):
                x = layer(route_connections.pop(), inputs.pop())
            else:
                x = layer(x)
        return outputs
    def _create_conv_layers(self):
        layers = []
        for module in neck:
            if isinstance(module, tuple):
                out_channels, kernel_size, strides, padding, act_func, name = module
                layers.append(
                    CNNBlock(
                        out_channels,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding=padding, activation=act_func))
                in_channels = out_channels
            elif isinstance(module, list):
                l = module[0]
                if l == 'M':
                    layers.append(max_pool())
                elif l == 'C':
                    layers.append(concat(in_channels))
            elif isinstance(module, str):
                    layers.append(tf.keras.layers.UpSampling2D(size=2))
        return layers



class ScalePrediction(tf.keras.layers.Layer):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv_1 = CNNBlock(2 * in_channels, kernel_size=3, padding='same')
        self.conv_2 = CNNBlock((num_classes + 5) * 3, bn_act=False, kernel_size=1)
        self.num_classes = num_classes
    def call(self, input_tensor):
        if input_tensor.get_shape()[1] == 52:
            x = self.conv_2(input_tensor)
            x = tf.reshape(x, shape=(K.shape(x)[0], K.shape(x)[1], K.shape(x)[2], 3, self.num_classes + 5))
            return x
        else:
            x = self.conv_1(input_tensor)
            x = self.conv_2(x)
            x = tf.reshape(x, shape=(K.shape(x)[0], K.shape(x)[1], K.shape(x)[2], 3, self.num_classes + 5))
            return x # TensorShape([3, 416, 416, 3, 85])


class concat_head(tf.keras.layers.Layer):
    def __init__(self, in_channels):
        super().__init__()
        self.ch = in_channels
        self.conc = tf.keras.layers.Concatenate(axis=-1)
        self.conv = CNNBlock(self.ch, kernel_size=1,strides=1,
                        padding='same', activation = 'leaky')
    def call(self, x_1, x_2):
        x = self.conc([x_1, x_2,])
        x = self.conv(x)
        return x


class yolov4_head(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.layers = self._create_conv_layers()
    def call(self, input_tensor):
        outputs = []  # for each scale
        inputs = input_tensor[:2]
        x = input_tensor[2]
        for i, layer in enumerate(self.layers):
            if i in [0, 7, 14]:
                outputs.append(layer(x))
                continue
            elif isinstance(layer, concat_head):
                x = layer(x, inputs.pop())
            else:
                x = layer(x)
        return outputs
    def _create_conv_layers(self):
        layers = []
        for module in head:
            if isinstance(module, tuple):
                out_channels, kernel_size, strides, padding, act_func, name = module
                layers.append(
                    CNNBlock(
                        out_channels,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding=padding, activation=act_func))
                in_channels = out_channels
            elif isinstance(module, list):
                l, c = module[0], module[1]
                if l == 'C':
                    layers.append(concat_head(in_channels=c))
                elif l == 'S':
                    layers.append(ScalePrediction(in_channels=c, num_classes= 20))
        return layers


def YoloV4(num_classes, shape=(416, 416, 3), training=True):
    inputs = Input(shape=shape)
    model_backbone = backbone_layers()
    out_backbone = model_backbone(inputs)
    model_neck = yolov4_neck()
    out_neck = model_neck(out_backbone)
    model_head = yolov4_head()
    y_small, y_medium, y_large = model_head(out_neck)
    return tf.keras.Model(inputs, (y_small, y_medium, y_large))


def get_absolute_yolo_box(y_pred, valid_anchors_wh, num_classes):
    """
    inputs:
    y_pred: Prediction tensor from the model output, in the shape of (batch, grid, grid, anchor, 5 + num_classes)
    outputs:
    y_box: boxes in shape of (batch, grid, grid, anchor, 4), the last dimension is (xmin, ymin, xmax, ymax)
    objectness: probability that an object exists
    classes: probability of classes
    """
    t_xy, t_wh, objectness, classes = tf.split(
        y_pred, (2, 2, 1, num_classes), axis=-1)
    objectness = tf.sigmoid(objectness)
    classes = tf.sigmoid(classes)
    grid_size = tf.shape(y_pred)[1]
    C_xy = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    C_xy = tf.stack(C_xy, axis=-1)
    C_xy = tf.expand_dims(C_xy, axis=2)  # [gx, gy, 1, 2]
    b_xy = tf.sigmoid(t_xy) + tf.cast(C_xy, tf.float32)
    b_xy = b_xy / tf.cast(grid_size, tf.float32)
    b_wh = tf.exp(t_wh) * valid_anchors_wh
    y_box = tf.concat([b_xy, b_wh], axis=-1)
    return y_box, objectness, classes


def get_relative_yolo_box(y_true, valid_anchors_wh):
    """
    This is the inverse of `get_absolute_yolo_box` above.
    """
    grid_size = tf.shape(y_true)[1]
    C_xy = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    C_xy = tf.expand_dims(tf.stack(C_xy, axis=-1), axis=2)
    b_xy = y_true[..., 0:2]
    b_wh = y_true[..., 2:4]
    t_xy = b_xy * tf.cast(grid_size, tf.float32) - tf.cast(C_xy, tf.float32)
    t_wh = tf.math.log(b_wh / valid_anchors_wh)
    # b_wh could have some cells are 0, divided by anchor could result in inf or nan
    t_wh = tf.where(
        tf.logical_or(tf.math.is_inf(t_wh), tf.math.is_nan(t_wh)),
        tf.zeros_like(t_wh), t_wh)
    y_box = tf.concat([t_xy, t_wh], axis=-1)
    return y_box


class YoloLoss(object):
    def __init__(self, num_classes, valid_anchors_wh):
        self.num_classes = num_classes
        self.ignore_thresh = 0.5
        self.valid_anchors_wh = valid_anchors_wh
        self.lambda_coord = 5.0
        self.lamda_noobj = 0.5
    def __call__(self, y_true, y_pred):
        """
        calculate the loss of model prediction for one scale
        """
        pred_xy_rel = tf.sigmoid(y_pred[..., 0:2])
        pred_wh_rel = y_pred[..., 2:4]
        pred_box_abs, pred_obj, pred_class = get_absolute_yolo_box(
            y_pred, self.valid_anchors_wh, self.num_classes)
        pred_box_abs = xywh_to_x1x2y1y2(pred_box_abs)

        true_xy_abs, true_wh_abs, true_obj, true_class = tf.split(
            y_true, (2, 2, 1, self.num_classes), axis=-1)
        true_box_abs = tf.concat([true_xy_abs, true_wh_abs], axis=-1)
        true_box_abs = xywh_to_x1x2y1y2(true_box_abs)

        true_box_rel = get_relative_yolo_box(y_true, self.valid_anchors_wh)
        true_xy_rel = true_box_rel[..., 0:2]
        true_wh_rel = true_box_rel[..., 2:4]

        weight = 2 - true_wh_abs[..., 0] * true_wh_abs[..., 1]

        xy_loss = self.calc_xy_loss(true_obj, true_xy_rel, pred_xy_rel, weight)
        wh_loss = self.calc_wh_loss(true_obj, true_wh_rel, pred_wh_rel, weight)
        class_loss = self.calc_class_loss(true_obj, true_class, pred_class)

        ignore_mask = self.calc_ignore_mask(true_obj, true_box_abs,
                                            pred_box_abs)
        obj_loss = self.calc_obj_loss(true_obj, pred_obj, ignore_mask)

        return xy_loss + wh_loss + class_loss + obj_loss, (xy_loss, wh_loss,
                                                           class_loss,
                                                           obj_loss)

    def calc_ignore_mask(self, true_obj, true_box, pred_box):
        # YOLOv3:
        true_box_shape = tf.shape(true_box)
        # (None, 13, 13, 3, 4)
        pred_box_shape = tf.shape(pred_box)
        # (None, 507, 4)
        true_box = tf.reshape(true_box, [true_box_shape[0], -1, 4])
        # sort true_box to have non-zero boxes rank first
        true_box = tf.sort(true_box, axis=1, direction="DESCENDING")
        # (None, 100, 4)
        true_box = true_box[:, 0:100, :]
        # (None, 507, 4)
        pred_box = tf.reshape(pred_box, [pred_box_shape[0], -1, 4])
        # (None, 507, 507)
        iou = broadcast_iou(pred_box, true_box)
        # (None, 507)
        best_iou = tf.reduce_max(iou, axis=-1)
        # (None, 13, 13, 3)
        best_iou = tf.reshape(best_iou, [pred_box_shape[0], pred_box_shape[1], pred_box_shape[2], pred_box_shape[3]])
        # ignore_mask = 1 => don't ignore
        # ignore_mask = 0 => should ignore
        ignore_mask = tf.cast(best_iou < self.ignore_thresh, tf.float32)
        # (None, 13, 13, 3, 1)
        ignore_mask = tf.expand_dims(ignore_mask, axis=-1)
        return ignore_mask

    def calc_obj_loss(self, true_obj, pred_obj, ignore_mask):
        """
        calculate loss of objectness: sum of L2 distances
        inputs:
        true_obj: objectness from ground truth in shape of (batch, grid, grid, anchor, num_classes)
        pred_obj: objectness from model prediction in shape of (batch, grid, grid, anchor, num_classes)
        outputs:
        obj_loss: objectness loss
        """
        obj_entropy = binary_cross_entropy(pred_obj, true_obj)

        obj_loss = true_obj * obj_entropy
        noobj_loss = (1 - true_obj) * obj_entropy * ignore_mask

        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3, 4))
        noobj_loss = tf.reduce_sum(
            noobj_loss, axis=(1, 2, 3, 4)) * self.lamda_noobj
        return obj_loss + noobj_loss

    def calc_class_loss(self, true_obj, true_class, pred_class):
        """
        calculate loss of class prediction
        inputs:
        true_obj: if the object present from ground truth in shape of (batch, grid, grid, anchor, 1)
        true_class: one-hot class from ground truth in shape of (batch, grid, grid, anchor, num_classes)
        pred_class: one-hot class from model prediction in shape of (batch, grid, grid, anchor, num_classes)
        outputs:
        class_loss: class loss
        """
        class_loss = binary_cross_entropy(pred_class, true_class)
        class_loss = true_obj * class_loss
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3, 4))
        return class_loss

    def calc_xy_loss(self, true_obj, true_xy, pred_xy, weight):
        """
        calculate loss of the centroid coordinate: sum of L2 distances
        inputs:
        true_obj: if the object present from ground truth in shape of (batch, grid, grid, anchor, 1)
        true_xy: centroid x and y from ground truth in shape of (batch, grid, grid, anchor, 2)
        pred_xy: centroid x and y from model prediction in shape of (batch, grid, grid, anchor, 2)
        weight: weight adjustment, reward smaller bounding box
        outputs:
        xy_loss: centroid loss
        """
        # shape (batch, grid, grid, anchor), eg. (32, 13, 13, 3)
        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)

        # in order to element-wise multiply the result from tf.reduce_sum
        # we need to squeeze one dimension for objectness here
        true_obj = tf.squeeze(true_obj, axis=-1)
        xy_loss = true_obj * xy_loss * weight
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3)) * self.lambda_coord
        return xy_loss

    def calc_wh_loss(self, true_obj, true_wh, pred_wh, weight):
        """
        calculate loss of the width and height: sum of L2 distances
        inputs:
        true_obj: if the object present from ground truth in shape of (batch, grid, grid, anchor, 1)
        true_wh: width and height from ground truth in shape of (batch, grid, grid, anchor, 2)
        pred_wh: width and height from model prediction in shape of (batch, grid, grid, anchor, 2)
        weight: weight adjustment, reward smaller bounding box
        outputs:
        wh_loss: width and height loss
        """
        # shape (batch, grid, grid, anchor), eg. (32, 13, 13, 3)
        wh_loss = tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        true_obj = tf.squeeze(true_obj, axis=-1)
        wh_loss = true_obj * wh_loss * weight
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3)) * self.lambda_coord
        return wh_loss