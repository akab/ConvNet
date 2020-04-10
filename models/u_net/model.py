import os
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt

from models.base import BaseNetwork, LossHistory, load_weights_from_dir


def get_unet(weights_path, model_path):
    if not os.path.exists(model_path):
        print('Cannot find U-Net model file')
        exit(-1)
    model = load_model(model_path)

    last_weights = load_weights_from_dir(weights_path)
    if last_weights is None:
        print('Cannot find weights file')
        exit(-1)
    model.load_weights(last_weights)

    return model


class UNet(BaseNetwork):
    def __init__(self, name='u_net', pretrained_weights=None, input_size=(64, 64, 1)):
        """
        :param pretrained_weights:
        :param input_size: should be 64 multiple
        """
        super(UNet, self).__init__(name, input_size)
        self.layer_dict = dict([])
        self.architecture = Model()
        self.define()
        if pretrained_weights is not None:
            self.architecture.load_weights(pretrained_weights)

    def define(self):
        """
        Define the architecture of U-NET Model
        """
        if K.image_data_format() == 'channels_first':
            input_shape = (1, self.input_size[0], self.input_size[1])
        else:
            input_shape = (self.input_size[0], self.input_size[1], 1)

        inputs = Input(input_shape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Dropout(0.2)(conv1)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Dropout(0.2)(conv2)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Dropout(0.2)(conv3)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        self.architecture = Model(inputs=inputs, outputs=conv10)

        self.architecture.compile(optimizer=SGD(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        self.architecture.save(r'models\u_net\unet.h5')

        self.architecture.summary()

        # get the symbolic outputs of each "key" layer (we gave them unique names).
        self.layer_dict = dict([(layer.name, layer) for layer in self.architecture.layers])

        # save the model as json string
        json_string = self.architecture.to_json()
        open(r'models\u_net\architecture.json', 'w').write(json_string)

        return self.architecture

    @staticmethod
    def plot_history(history):
        # Plot training & validation accuracy values
        plt.plot(history.history['acc'])
        # plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(r'models/u_net/results/train_accuracy.png')

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(r'models/u_net/results/train_loss.png')

    def train(self, train_gen):
        """
        :param train_gen:
        :return:
        """
        model_checkpoint = ModelCheckpoint(r'models/u_net/checkpoints/weights.{epoch:02d}-{loss:.2f}.hdf5',
                                           monitor='loss', mode='auto',
                                           verbose=1, save_best_only=True)
        history_cb = LossHistory()
        history_report = self.architecture.fit_generator(train_gen, steps_per_epoch=32, epochs=150, verbose=1,
                                                         callbacks=[model_checkpoint, history_cb])
        self.plot_history(history_report)

    def validate(self, test_gen):
        """
        :param test_gen:
        :return:
        """
        results = self.architecture.predict_generator(generator=test_gen, steps=30, verbose=1)
        for i, item in enumerate(results):
            pil_img = array_to_img(item)
            pil_img.save(os.path.abspath(r'models/u_net/results/prediction') + "/" + str(i) + "_predict.bmp")
            print('Saving', i, '-nth prediction...')
