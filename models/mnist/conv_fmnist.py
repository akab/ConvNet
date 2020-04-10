import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from keras.utils import to_categorical

logdir = 'fashionMNIST-logs'
embed_count = 2500

# Creating the embedding variable with all the images defined above under X_test
(_, _), (X_test, Y_test) = fashion_mnist.load_data()

X_test = np.array(X_test[0:embed_count])
Y_test = np.array(Y_test[0:embed_count])

embedding_var = tf.Variable(X_test, name='fmnist_embedding')

# We add only one embedding
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name

# Link to tensor labels
embedding.metadata_path = os.path.join(logdir, 'metadata.tsv')
# Create logger
summary_writer = tf.summary.FileWriter(logdir=logdir)

# Writer projector_config.pbtxt in the logdir.
# TensorBoard will read this file during startup.
projector.visualize_embeddings(summary_writer=summary_writer, config=config)

# Periodically save the model variables in a checkpoint in logdir.
with tf.Session() as s:
    s.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(s, os.path.join(logdir, 'model.ckpt'))

# Create the sprite image
rows = 28
cols = 28
label = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

sprite_dim = int(np.sqrt(X_test.shape[0]))
sprite_image = np.ones((cols * sprite_dim, rows * sprite_dim))

index = 0
labels = []
for i in range(sprite_dim):
    for j in range(sprite_dim):
        labels.append(label[int(Y_test[index])])

        sprite_image[
        i * cols: (i + 1) * cols,
        j * rows: (j + 1) * rows
        ] = X_test[index].reshape(28, 28) * -1 + 1

        index += 1

# After constructing the sprite, I need to tell the Embedding Projector where to find it
embedding.sprite.image_path = os.path.join(logdir, 'sprite.png')
embedding.sprite.single_image_dim.extend([28, 28])

# Create the metadata (labels) file
with open(embedding.metadata_path, 'w') as meta:
    meta.write('Index\tLabel\n')
    for index, label in enumerate(labels):
        meta.write('{}\t{}\n'.format(index, label))

plt.imsave(embedding.sprite.image_path, sprite_image, cmap='gray')
plt.imshow(sprite_image, cmap='gray')
plt.show()
