import os
from tf_handler import TFHandler
from data_factory import DataFactory
from models.u_net.model import UNet
from models.VisualizeMaxFeatureMap import VisualizeMaxFeatureMap
from models.base import load_weights_from_dir

input_size = (212, 180)
target_size = (256, 256)

classes = None  # ['bru', 'icl', 'ins', 'pelo', 'pie', 'spo', 'str']
checkpoints_dir = r'models\u_net\checkpoints'
train_dir = r'data\linescan_woclass\train'  # r'data\DRIVE\training'
test_dir = r'data\linescan_woclass\test\image'  # r'data\DRIVE\test'
visualize_activations = False
num_classes = 1
data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')


def write_conv_layers(model, pich_shape):
    assert len(pich_shape) == 3

    visualizer = VisualizeMaxFeatureMap(pic_shape=pich_shape, model=model)
    print("Find images that maximize feature maps")
    visualizer.find_images(input_img=None, layer_names=None, max_nfmap=512,
                           save_dir=os.path.join('models', model.name, 'results', 'conv_layers'))
    print("Plot images...")


def unet():
    # Setup TensorFlow Backend
    tf_handler = TFHandler()
    tf_handler.configure()

    # Prepare data
    data_factory = DataFactory(classes=classes, image_dir=train_dir + r"\image", label_dir=train_dir + r"\label",
                               test_dir=test_dir,
                               input_size=input_size)

    # Load pre-trained weights, if any
    last_weights = load_weights_from_dir(checkpoints_dir)

    # Define, train and validate model
    net = UNet(input_size=target_size, pretrained_weights=last_weights)

    net.train(data_factory.train_generator(batch_size=2, aug_dict=data_gen_args, target_size=target_size,
                                           num_class=num_classes, save_to_dir=None))

    net.validate(data_factory.validation_generator(batch_size=2, aug_dict=data_gen_args, target_size=target_size))

    # Visualize activations
    if visualize_activations:
        write_conv_layers(model=net, pich_shape=(256, 256, 1))

    # Dealloc TensorFlow
    tf_handler.close()
