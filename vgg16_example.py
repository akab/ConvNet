import os

from tf_handler import TFHandler
from models.VGG16.model import VGG16
from models.VisualizeMaxFeatureMap import VisualizeMaxFeatureMap


def vgg16_example():
    # Setup TensorFlow Backend
    tf_handler = TFHandler()
    tf_handler.configure()

    # Define model
    net = VGG16((128, 128))

    # Visualize activations
    visualizer = VisualizeMaxFeatureMap(pic_shape=None, model=net)
    print("Find images that maximize feature maps")
    argimage = visualizer.find_images(input_img=None,
                                      layer_names=['block5_conv1'],
                                      max_nfmap=512,
                                      save_dir=os.path.join('models', net.name, 'results', 'conv_layers'))

    print("Plot images...")
    visualizer.plot_images_wrapper(argimage)

    # Dealloc TensorFlow
    tf_handler.close()
