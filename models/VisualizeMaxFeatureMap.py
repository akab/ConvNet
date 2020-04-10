import os
import numpy as np
from PIL import Image
from keras import backend as K


class VisualizeMaxFeatureMap:
    def __init__(self, model, pic_shape=None):
        """
        :param pic_shape: dimension of a single picture e.g., (96,96,1)
        :param model: the model to work with
        """
        if pic_shape is None:
            self.pic_shape = (96, 96, 3)
        else:
            self.pic_shape = pic_shape
        self.model = model

    @staticmethod
    def create_iterate(input_img, layer_output, filter_index):
        """
        layer_output[:,:,:,0] is (Nsample, 94, 94) tensor contains:
        W0^T [f(image)]_{i,j}], i = 1,..., 94, j = 1,..., 94

        layer_output[:,:,:,1] contains:
        W1^T [f(image)]_{i,j}], i = 1,..., 94, j = 1,..., 94

        W0 and W1 are different kernel!
        :param input_img:
        :param layer_output:
        :param filter_index:
        :return: the iteration function
        """
        # loss is a scalar
        if len(layer_output.shape) == 4:
            # conv layer
            loss = K.mean(layer_output[:, :, :, filter_index])
        elif len(layer_output.shape) == 2:
            # fully connected layer
            loss = K.mean(layer_output[:, filter_index])

        # calculate the gradient of the loss evaluated at the provided image
        grads = K.gradients(loss, input_img)[0]

        # normalize the gradients by its L2 norm
        grads /= (K.sqrt(K.mean(K.square(grads))) + K.epsilon())

        # iterate is a function taking (input_img, scalar) and output [loss_value, gradient_value]
        iterate = K.function([input_img, K.learning_phase()], [loss, grads])

        return iterate

    @staticmethod
    def deprocess_image(x):
        """
        util function to convert a tensor into a valid image
        :param x:
        :return:
        """
        # normalize tensor: center on 0., ensure std is 0.1
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        # Shift x to have a mean 0.5 and std 0.1
        # This means 95% of the x should be in between 0 and 1
        # if x is normal
        x += 0.5
        x = np.clip(x, 0, 1)

        # rescale values to range between 0 and 255
        x *= 255
        if K.image_data_format() == 'channels_first':
            x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')

        return x

    def find_n_feature_map(self, layer_name, max_nfmap):
        """
        shows the number of feature maps for this layer
        (only works if the layer is CNN)
        :param layer_name:
        :param max_nfmap: the maximum number of feature map to be used for each layer.
        :return:
        """
        if layer_name in self.model.layer_dict:
            weights = self.model.layer_dict[layer_name].get_weights()
            n_fmap = weights[1].shape[0]
        else:
            print(layer_name + " is not one of the layer names..")
            n_fmap = 1
        n_fmap = np.min([max_nfmap, n_fmap])

        return int(n_fmap)

    def find_image_maximizing_activation(self, iterate,
                                         picorig=False,
                                         n_iter=30):
        """
        The input image is scaled to range between 0 and 1
        :param iterate:
        :param picorig: True if the picture image for input is original scale
                         ranging between 0 and 225
                   False if the picture image for input is ranging [0,1]
        :param n_iter:
        :return:
        """

        input_img_data = np.random.random((1,
                                           self.pic_shape[0],
                                           self.pic_shape[1],
                                           self.pic_shape[2]))
        if picorig:
            # if the original picture is unscaled and ranging between (0,225),
            # then the image values are centered around 123 with STD=25
            input_img_data = input_img_data * 25 + 123
        # I played with this step value but the final image looks to be robust
        step = 500

        # gradient ascent
        loss_values = []
        for i in range(n_iter):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step
            print('(iter #', i, ') Current loss value:%d' % loss_value)
            if loss_value <= 0.:
                # some filters get stuck to 0, we can skip them
                print('Loss is 0, skipping...')
                break
            loss_values.append(loss_value)
        return input_img_data, loss_values

    def find_images(self, input_img, max_nfmap, save_dir, layer_names=None, picorig=False, n_iter=30):
        """
        Search images that maximizes feature maps
        :param input_img:the alias of the input layer from the deep learning model
        :param max_nfmap: the maximum number of feature map to be used for each layer.
        :param layer_names: list containing the name of the layers whose feature maps to be used
                            (if 'None' compute for all conv layers
        :param save_dir: path where save images
        :param picorig:
        :param n_iter:
        :return: dictionary
                            key = layer name
                            value = a list containing the tuple of (images, list of loss_values)
                                    that maximize each feature map
        """
        argimage = {}
        if input_img is None:
            input_img = self.model.architecture.input
        if layer_names is None:
            layer_names = [layer_name for layer_name in self.model.layer_dict if "conv" in layer_name]

        # Look for the image for each feature map of each layer one by one
        for layer_name in layer_names:
            print('Initializing layer %s...' % layer_name)
            n_fmap = self.find_n_feature_map(layer_name, max_nfmap)
            layer_output = self.model.layer_dict[layer_name].output
            result = self.find_images_for_layer(layer_name=layer_name,
                                                input_img=input_img,
                                                layer_output=layer_output,
                                                indices=range(n_fmap),
                                                n_iter=n_iter)

            argimage[layer_name] = result
            self.write_image(argimage=argimage, layer_name=layer_name, save_dir=save_dir)
        return argimage

    def find_images_for_layer(self, layer_name, input_img, layer_output, indices, picorig=False, n_iter=30):
        """
        :param layer_name:
        :param input_img:
        :param layer_output:
        :param indices: list containing index of
                            --> filtermaps of CNN or
                            --> nodes of fully-connected layer
        :param picorig:
        :param n_iter:
        :return: a list containing the tuple of (images, list of loss_values)
                that maximize each feature map
        """
        result_temp = []
        for filter_index in indices:  # filtermap to visualize
            print('Processing filter', filter_index, '/', indices.__len__(), 'of layer', layer_name)
            iterate = self.create_iterate(input_img=input_img, layer_output=layer_output, filter_index=filter_index)
            input_img_data, loss_values = self.find_image_maximizing_activation(iterate, picorig=picorig, n_iter=n_iter)
            result_temp.append((input_img_data, loss_values))

        return result_temp

    def write_image(self, argimage, layer_name, save_dir):
        print('Writing conv image', layer_name, '...')

        dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, save_dir))
        os.makedirs(dir, exist_ok=True)

        layer_dir = dir + "/" + layer_name
        os.makedirs(layer_dir, exist_ok=True)
        n_fmap = len(argimage[layer_name])
        for value, i in zip(argimage[layer_name], range(n_fmap)):
            print('Writing filter image', i, 'for layer', layer_name, '...')
            input_img_data = value[0][0]

            # img = self.deprocess_image(input_img_data)
            # pil_img = Image.fromarray(img.astype(np.uint8))
            from keras.preprocessing.image import array_to_img
            pil_img = array_to_img(input_img_data)
            pil_img.save(layer_dir + "/n_featuremap=" + str(n_fmap) + "_filter=" + str(i) + ".bmp")

    def plot_images_wrapper(self, argimage):
        """
        :param argimage:
        :return:
        """
        print('Writing conv images...')
        if argimage.keys().__len__() > 1:
            layer_names = sorted(argimage.keys())
        else:
            layer_names = argimage.keys()

        dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, 'results', 'unet'))  # 'VGG16_layer_vis'
        os.makedirs(dir, exist_ok=True)

        for layer_name in layer_names:
            layer_dir = dir + "/" + layer_name
            os.makedirs(layer_dir, exist_ok=True)
            n_fmap = len(argimage[layer_name])
            for value, i in zip(argimage[layer_name], range(n_fmap)):
                print('Writing filter image', i, 'for layer', layer_name, '...')
                input_img_data = value[0][0]

                # img = self.deprocess_image(input_img_data)
                # pil_img = Image.fromarray(img.astype(np.uint8))
                from keras.preprocessing.image import array_to_img
                pil_img = array_to_img(input_img_data)
                pil_img.save(layer_dir + "/n_featuremap=" + str(n_fmap) + "_filter=" + str(i) + ".bmp")
