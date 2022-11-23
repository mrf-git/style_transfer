Image Style Transfer
====================

Implements image style transfer based on the method described by Gatys et al.

See references for details:
* [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).
Gatys et al.

* [Preserving Color in Neural Artistic Style Transfer](https://arxiv.org/pdf/1606.05897.pdf). Gatys et al.


## Requirements
* numpy/scipy
* Tensorflow 2.1
* imageio

## Usage

#### Preparing the VGG16 model

The style transfer script requires the VGG16 model to be prepared first. To do this, run the `export_vgg16.py` script:

    >> python export_vgg16.py /path/to/model.bin

Replace `/path/to/model.bin` with the appropriate filename to save to model.

The first time running this, Tensorflow will download the VGG16 model weights from the Internet.
Once the model has been prepared, you do not need to run this script again. The saved model will require approximately
335 MB of space.

#### Transferring styles
To transfer image styles, run the `transfer_style.py` script:

    >> python transfer_style.py /path/to/model.bin content_image.jpg style_image.jpg output_image.jpg

This will transfer the style from "style_image.jpg" onto "content_image.jpg" and write the result to "output_image.jpg".
See the help for more options:

    >> python transfer_style.py --help

## License
MIT license.

## Examples

Original image:

![](examples/original1.jpg?raw=True)

Style to transfer:

![](examples/style1.jpg?raw=True)

Result:

![](examples/out1.jpg?raw=True)

Result preserving color:

![](examples/out2.jpg?raw=True)

Other example styles for the following images:

![](examples/original2.jpg?raw=True)


![](examples/out3.jpg?raw=True) ![](examples/out4.jpg?raw=True) ![](examples/out5.jpg?raw=True) ![](examples/out6.jpg?raw=True)
