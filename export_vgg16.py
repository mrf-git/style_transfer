"""
Loads the VGG16 model from Tensorflow-Keras and builds a new Keras model for computing style and content features for
image style transfer. The model is modified to allow input images of arbitrary size, weights are converted to
constants, and all max pooling layers are convered to average pooling layers.

See the paper for details on the VGG architecture:
* Very deep convolutional networks for large-scale image recognition. 2015. K. Simonyan and A. Zisserman.
"""

import argparse
import os
import pickle
import tensorflow as tf

from gram import GramLayer


def main(output_filename):
    vgg_model = tf.keras.applications.vgg16.VGG16(weights="imagenet")

    input_layer = tf.keras.layers.Input(shape=[None, None, 3], dtype=tf.float32, name="vgg_feats_input")
    layer = tf.keras.layers.Subtract()([input_layer,
                                        tf.constant([123.68, 116.78, 103.94], dtype=tf.float32)[None, None, None, :]])
    layer = layer[:, :, :, ::-1]

    vgg_layer = vgg_model.get_layer(index=1)
    block1_conv1 = tf.keras.layers.Conv2D(64, 3, activation="relu", trainable=False, name="block1_conv1", padding="same",
                                          kernel_initializer=tf.constant_initializer(vgg_layer.weights[0].numpy()),
                                          bias_initializer=tf.constant_initializer(vgg_layer.weights[1].numpy()))
    layer_1_1 = block1_conv1(layer)

    vgg_layer = vgg_model.get_layer(index=2)
    block1_conv2 = tf.keras.layers.Conv2D(64, 3, activation="relu", trainable=False, name="block1_conv2", padding="same",
                                          kernel_initializer=tf.constant_initializer(vgg_layer.weights[0].numpy()),
                                          bias_initializer=tf.constant_initializer(vgg_layer.weights[1].numpy()))
    layer_1_2 = block1_conv2(layer_1_1)
    pooled = tf.keras.layers.AveragePooling2D(name="block1_pool")(layer_1_2)

    vgg_layer = vgg_model.get_layer(index=4)
    block2_conv1 = tf.keras.layers.Conv2D(128, 3, activation="relu", trainable=False, name="block2_conv1", padding="same",
                                          kernel_initializer=tf.constant_initializer(vgg_layer.weights[0].numpy()),
                                          bias_initializer=tf.constant_initializer(vgg_layer.weights[1].numpy()))
    layer_2_1 = block2_conv1(pooled)

    vgg_layer = vgg_model.get_layer(index=5)
    block2_conv2 = tf.keras.layers.Conv2D(128, 3, activation="relu", trainable=False, name="block2_conv2", padding="same",
                                          kernel_initializer=tf.constant_initializer(vgg_layer.weights[0].numpy()),
                                          bias_initializer=tf.constant_initializer(vgg_layer.weights[1].numpy()))
    layer_2_2 = block2_conv2(layer_2_1)
    pooled = tf.keras.layers.AveragePooling2D(name="block2_pool")(layer_2_2)

    vgg_layer = vgg_model.get_layer(index=7)
    block3_conv1 = tf.keras.layers.Conv2D(256, 3, activation="relu", trainable=False, name="block3_conv1", padding="same",
                                          kernel_initializer=tf.constant_initializer(vgg_layer.weights[0].numpy()),
                                          bias_initializer=tf.constant_initializer(vgg_layer.weights[1].numpy()))
    layer_3_1 = block3_conv1(pooled)

    vgg_layer = vgg_model.get_layer(index=8)
    block3_conv2 = tf.keras.layers.Conv2D(256, 3, activation="relu", trainable=False, name="block3_conv2", padding="same",
                                          kernel_initializer=tf.constant_initializer(vgg_layer.weights[0].numpy()),
                                          bias_initializer=tf.constant_initializer(vgg_layer.weights[1].numpy()))
    layer_3_2 = block3_conv2(layer_3_1)

    vgg_layer = vgg_model.get_layer(index=9)
    block3_conv3 = tf.keras.layers.Conv2D(256, 3, activation="relu", trainable=False, name="block3_conv3", padding="same",
                                          kernel_initializer=tf.constant_initializer(vgg_layer.weights[0].numpy()),
                                          bias_initializer=tf.constant_initializer(vgg_layer.weights[1].numpy()))
    layer_3_3 = block3_conv3(layer_3_2)
    pooled = tf.keras.layers.AveragePooling2D(name="block3_pool")(layer_3_3)

    vgg_layer = vgg_model.get_layer(index=11)
    block4_conv1 = tf.keras.layers.Conv2D(512, 3, activation="relu", trainable=False, name="block4_conv1", padding="same",
                                          kernel_initializer=tf.constant_initializer(vgg_layer.weights[0].numpy()),
                                          bias_initializer=tf.constant_initializer(vgg_layer.weights[1].numpy()))
    layer_4_1 = block4_conv1(pooled)

    vgg_layer = vgg_model.get_layer(index=12)
    block4_conv2 = tf.keras.layers.Conv2D(512, 3, activation="relu", trainable=False, name="block4_conv2", padding="same",
                                          kernel_initializer=tf.constant_initializer(vgg_layer.weights[0].numpy()),
                                          bias_initializer=tf.constant_initializer(vgg_layer.weights[1].numpy()))
    layer_4_2 = block4_conv2(layer_4_1)

    vgg_layer = vgg_model.get_layer(index=13)
    block4_conv3 = tf.keras.layers.Conv2D(512, 3, activation="relu", trainable=False, name="block4_conv3", padding="same",
                                          kernel_initializer=tf.constant_initializer(vgg_layer.weights[0].numpy()),
                                          bias_initializer=tf.constant_initializer(vgg_layer.weights[1].numpy()))
    layer_4_3 = block4_conv3(layer_4_2)
    pooled = tf.keras.layers.AveragePooling2D(name="block4_pool")(layer_4_3)

    vgg_layer = vgg_model.get_layer(index=15)
    block5_conv1 = tf.keras.layers.Conv2D(512, 3, activation="relu", trainable=False, name="block5_conv1", padding="same",
                                          kernel_initializer=tf.constant_initializer(vgg_layer.weights[0].numpy()),
                                          bias_initializer=tf.constant_initializer(vgg_layer.weights[1].numpy()))
    layer_5_1 = block5_conv1(pooled)

    vgg_layer = vgg_model.get_layer(index=16)
    block5_conv2 = tf.keras.layers.Conv2D(512, 3, activation="relu", trainable=False, name="block5_conv2", padding="same",
                                          kernel_initializer=tf.constant_initializer(vgg_layer.weights[0].numpy()),
                                          bias_initializer=tf.constant_initializer(vgg_layer.weights[1].numpy()))
    layer_5_2 = block5_conv2(layer_5_1)

    vgg_layer = vgg_model.get_layer(index=17)
    block5_conv3 = tf.keras.layers.Conv2D(512, 3, activation="relu", trainable=False, name="block5_conv3", padding="same",
                                          kernel_initializer=tf.constant_initializer(vgg_layer.weights[0].numpy()),
                                          bias_initializer=tf.constant_initializer(vgg_layer.weights[1].numpy()))
    layer_5_3 = block5_conv3(layer_5_2)

    content_layers = [tf.keras.layers.Flatten()(layer_1_1), tf.keras.layers.Flatten()(layer_1_2),
                      tf.keras.layers.Flatten()(layer_2_1), tf.keras.layers.Flatten()(layer_2_2),
                      tf.keras.layers.Flatten()(layer_3_1), tf.keras.layers.Flatten()(layer_3_2),
                      tf.keras.layers.Flatten()(layer_3_3), tf.keras.layers.Flatten()(layer_4_1),
                      tf.keras.layers.Flatten()(layer_4_2), tf.keras.layers.Flatten()(layer_4_3),
                      tf.keras.layers.Flatten()(layer_5_1), tf.keras.layers.Flatten()(layer_5_2),
                      tf.keras.layers.Flatten()(layer_5_3)]

    gram_layers = [GramLayer()(layer_1_2), GramLayer()(layer_2_2), GramLayer()(layer_3_3),
                   GramLayer()(layer_4_3), GramLayer()(layer_5_3)]

    content_output = tf.keras.layers.Concatenate()(content_layers)
    style_output = tf.keras.layers.Concatenate()(gram_layers)

    model = tf.keras.Model([input_layer], [content_output, style_output])

    with open(output_filename, "wb") as f_out:
        pickle.dump(model.to_json(), f_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("output_filename", help="Path to store the saved model")

    args = parser.parse_args()
    main(os.path.abspath(args.output_filename))
