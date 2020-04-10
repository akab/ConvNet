from models.base import BaseNetwork

from keras.applications import vgg16


class VGG16(BaseNetwork):
    def __init__(self, input_size=None, pretrained_weights=None):
        super(VGG16, self).__init__(pretrained_weights, input_size)

        self.architecture = self.define()
        # this is the placeholder for the input images
        self.input_img = self.architecture.input
        # get the symbolic outputs of each "key" layer (we gave them unique names).
        self.layer_dict = dict([(layer.name, layer) for layer in self.architecture.layers[1:]])

    def define(self):
        # build the VGG16 network with ImageNet weights
        self.architecture = vgg16.VGG16(weights='imagenet', include_top=False)
        print('Model loaded.')

        self.architecture.summary()

        return self.architecture

    def train(self, train):
        print("This is a pre-trained model!")
        return False

    def validate(self, test):
        print("This is a pre-trained model!")
        return False
