import numpy as np
np.random.seed(1)

import tensorflow as tf
tf.random.set_seed(1)

import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split # pip install and conda install scikit-learn

# get data
images = np.load('data/mitdb_images_np_28x28.npy')
images = images / 255

labels = np.load('data/mitdb_label.npy')

classes =['N','L','R','V','/','A','F','f','j','a','E','J','e','Q','S']

# replace class characters with indices 
index = 0 
for x in classes:
   for i in range(len(labels)):
       if labels[i] == x:
           labels[i] = index
   index = index +1

labels = np.int8(labels)

# split into train and test sets, stratify on labels to make sure all classes represented in both
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, 
                                                                        stratify=labels) #, random_state=5

# add a channel dimension
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

train_labels = tf.one_hot(train_labels, 10)
test_labels = tf.one_hot(test_labels, 10)

# create model architecture
model = models.Sequential()
model.add(layers.Conv2D(1, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.Flatten())
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))


# save model
model.save('results/baseline-model', save_format='tf')

# plot accuracies
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.savefig('results/baseline-model-accuracy.png')

## uncomment to load and use saved model
# model = tf.keras.models.load_model('results/baseline-model')
# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
# print(test_acc)