import numpy as np
from keras.preprocessing.image import ImageDataGenerator


class DataFactory:
    def __init__(self, classes, image_dir, label_dir, test_dir, input_size=(256, 256)):
        """
        Init quantities and parse the data
        :param image_dir: train set directory
        :param test_dir: test set directory
        :param input_size: the minimum quantity processable by net
        :return:
        """
        self.classes = classes
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.test_dir = test_dir
        self.input_size = input_size

    @staticmethod
    def adjust_data(img, num_class, mask=None):
        """
        adjust img/mask in range [0,1]
        :param img:
        :param mask:
        :param num_class:
        :return:
        """
        if np.max(img) > 1.0:
            img = img / 255

        if mask is not None:
            if num_class > 2:
                mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]
                new_mask = np.zeros(mask.shape + (num_class,))
                for i in range(num_class):
                    # for one pixel in the image, find the class in mask and convert it into one-hot vector
                    new_mask[mask == i, i] = 1
                new_mask = np.reshape(new_mask,
                                      (new_mask.shape[0], new_mask.shape[1] * new_mask.shape[2], new_mask.shape[3]))
                mask = new_mask

            elif np.max(mask) > 1.0:
                mask = mask / 255
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0

        return img, mask if mask is not None else img

    def train_generator(self, batch_size, aug_dict, num_class, image_color_mode="grayscale",
                        mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                        save_to_dir=None, target_size=(256, 256), seed=1):
        """
        Can generate image and mask at the same time use the same seed for image_datagen and mask_datagen
        to ensure the transformation for image and mask is the same.
        if you want to visualize the results of generator, set save_to_dir = "your path"
        :param batch_size: Size of the batches of data (default: 32).
        :param aug_dict: arguments for data augmentation
        :param image_color_mode: "rgb", "rgba" or "grayscale"
        :param mask_color_mode: "rgb", "rgba" or "grayscale"
        :param image_save_prefix: prefix to use for file names of saved pictures
        :param mask_save_prefix: prefix to use for file names of saved masks
        :param num_class: number of classes to train
        :param save_to_dir: dir to save training images
        :param target_size: (height, width) the dimensions to which all images found will be re-sized.
        :param seed: random seed for shuffling and transformations
        :return: image generator
        """
        image_datagen = ImageDataGenerator(**aug_dict)
        mask_datagen = ImageDataGenerator(**aug_dict)
        image_generator = image_datagen.flow_from_directory(
            directory=self.image_dir,
            classes=self.classes,
            class_mode=None,  # i.e. 2D one-hot encoded labels
            color_mode=image_color_mode,
            target_size=target_size,
            batch_size=batch_size,
            save_to_dir=save_to_dir,
            save_prefix=image_save_prefix,
            seed=seed
        )

        mask_generator = mask_datagen.flow_from_directory(
            directory=self.label_dir,
            classes=self.classes,
            class_mode=None,
            color_mode=mask_color_mode,
            target_size=target_size,
            batch_size=batch_size,
            save_to_dir=save_to_dir,
            save_prefix=mask_save_prefix,
            seed=seed
        )

        train_generator = zip(image_generator, mask_generator)
        for (img, mask) in train_generator:
            img, mask = self.adjust_data(img, num_class, mask)
            yield (img, mask)

    def validation_generator(self, batch_size, aug_dict, target_size, image_color_mode="grayscale"):
        """

        :param classes:
        :param image_color_mode:
        :param batch_size:
        :param aug_dict:
        :param target_size:
        :return:
        """
        aug_dict['rescale'] = 1. / 255
        test_datagen = ImageDataGenerator(**aug_dict)
        validation_generator = test_datagen.flow_from_directory(
            self.test_dir,
            classes=self.classes,
            color_mode=image_color_mode,
            target_size=target_size,
            batch_size=batch_size,
            class_mode=None)
        for img in validation_generator:
            yield img
