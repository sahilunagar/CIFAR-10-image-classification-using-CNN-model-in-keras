import numpy as np
from scipy.misc import imread, imresize
from keras.models import model_from_json

# Load model
model_architecture = 'cifar10_network.json'
model_weights = 'cifar10_weights.h5'
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)

# Load images
img_names = ['cat.jpg', 'dog.jpg']
imgs = [imresize(imread(img_name), (32,32)).astype('float32') for img_name in img_names]
imgs = np.array(imgs) / 255

# Train
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Predict
predictions = model.predict(imgs)
preds = np.argmax(predictions, axis=1)
print(preds)
