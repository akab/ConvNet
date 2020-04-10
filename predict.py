import sys
import os
from keras.preprocessing.image import array_to_img
import cv2
import numpy as np

from models.u_net.model import get_unet

if not os.path.exists(sys.argv[1]):
    print('Cannot find specified file')
    exit(-1)

x = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
x = cv2.resize(x, (256, 256))
x = np.expand_dims(x, axis=0)
x = np.expand_dims(x, axis=3)

model = get_unet(r'C:\Users\WI6\source\repos\convnet\models\u_net\best_weights\sgd_0.001',
                 r"C:\Users\WI6\source\repos\convnet\models\u_net\unet.h5")

prediction = model.predict(x)

pil_img = array_to_img(np.reshape(prediction, (256, 256, 1)))
pil_img.save(os.path.abspath(r'models/u_net/results/prediction/prediction.bmp'))
