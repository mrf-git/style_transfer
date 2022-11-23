"""
Transfers style from the specified style image onto the specified content image.
"""

import argparse
import imageio
import numpy as np
import os
import pickle
import tensorflow as tf

from scipy.optimize import fmin_l_bfgs_b

from color_transfer import transfer_color_histogram_matching, transfer_color_luminance
from gram import GramLayer


def opt_func(x, model, target_content, target_style, target_shape, content_weight, variation_weight):
    """Optimization function for the scipy L-BFGS optimizer."""

    with tf.GradientTape() as grad_tape:
        image_input = tf.constant(x.astype("float32").reshape(target_shape))
        grad_tape.watch(image_input)
        cur_content, cur_style = model(image_input)

        content_loss = tf.reduce_mean((cur_content - target_content)**2)
        style_loss = tf.reduce_sum((cur_style - target_style)**2)
        variation_loss = tf.image.total_variation(image_input)

        loss = content_weight * content_loss + (1-content_weight) * style_loss + variation_weight * variation_loss
        grads = grad_tape.gradient(loss, [image_input])[0]

    return loss.numpy()[0], grads.numpy().astype("float64").flatten()



def main(vgg_model_filename, content_image, style_image, output_image, preserve_color, content_weight,
         variation_weight, color_mix, max_iters, init_random):

    with open(vgg_model_filename, "rb") as f_in:
        model_json = pickle.load(f_in)

    model = tf.keras.models.model_from_json(model_json, custom_objects={"GramLayer": GramLayer})

    content_image_batch = np.expand_dims(imageio.imread(content_image, pilmode="RGB").astype("float32"), axis=0)
    style_image_batch = np.expand_dims(imageio.imread(style_image, pilmode="RGB").astype("float32"), axis=0)
    target_shape = content_image_batch.shape
    if init_random:
        x_init = np.random.random(len(content_image_batch.flatten())).astype("float64")
    else:
        x_init = content_image_batch.flatten().astype("float64")

    if preserve_color:
        style_image_batch[0] = transfer_color_histogram_matching(content_image_batch[0], style_image_batch[0])

    target_content, _ = model(content_image_batch)
    _, target_style = model(style_image_batch)

    out_image_batch, _, _ = fmin_l_bfgs_b(opt_func, x_init, args=(model, target_content, target_style, target_shape,
                                                                  content_weight, variation_weight),
                                          bounds=[(0., 255.)] * len(x_init), maxiter=max_iters, disp=10, factr=10.)

    out_image_batch = out_image_batch.reshape(target_shape)

    if preserve_color:
        lum_trans_out_image_batch = transfer_color_luminance(content_image_batch[0], out_image_batch[0])
        out_image_batch[0] = color_mix * lum_trans_out_image_batch + (1-color_mix) * out_image_batch[0]

    imageio.imwrite(output_image, out_image_batch[0].astype("uint8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("vgg_model_filename", help="Path to the file storing the saved VGG16 model to use")
    parser.add_argument("content_image", help="Filename of the image to use for content features")
    parser.add_argument("style_image", help="Filename of the image to use for style features")
    parser.add_argument("output_image", help="Filename of the style-transferred image to output")

    parser.add_argument("-c", action="store_true", help="Specify to preserve content image colors")
    parser.add_argument("--rand", action="store_true", help="Specify to initialize from random noise instead of content")

    parser.add_argument("--content_weight", type=float, default=0.1, help="Weight of content loss between 0-1 (Default: 0.1)")
    parser.add_argument("--variation_weight", type=float, default=0.01, help="Weight of image variation loss (Default: 0.01)")

    parser.add_argument("--color_mix", type=float, default=0.6, help="Weight of content color if color preserved, between 0-1 (Default: 0.6)")

    parser.add_argument("--max_iters", type=int, default=20, help="Maximum number of L-BFGS iterations (Default: 20)")

    args = parser.parse_args()
    main(os.path.abspath(args.vgg_model_filename), os.path.abspath(args.content_image), os.path.abspath(args.style_image),
         os.path.abspath(args.output_image), args.c, args.content_weight, args.variation_weight, args.color_mix,
         args.max_iters, args.rand)

