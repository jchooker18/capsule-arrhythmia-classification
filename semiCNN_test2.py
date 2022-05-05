from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Softmax, Dropout, Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras import losses
from tensorflow.keras import Sequential
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # pip install and conda install scikit-learn
from keras.datasets import mnist
import numpy as np


### build classifier
classifier = Sequential()
classifier.add(Conv2D(32, kernel_size=(1, 3), activation='relu',kernel_regularizer=regularizers.l2(l=0.01), input_shape=(28, 28, 1)))

classifier.add(Conv2D(64, kernel_size=(1, 4), activation='relu'))
classifier.add(AveragePooling2D(pool_size=(1, 2)))

# classifier.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# classifier.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Dropout(0.3))
classifier.add(Flatten())
classifier.add(Dense(128, activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(Dense(10, activation='softmax'))

classifier.build()

### get data
images = np.load('data/mitdb_images_np.npy')
images = images / 255

labels = np.load('data/mitdb_label.npy')

classes =['N','L','R','V','/','A','F','f','j','a','E','J','e','Q','S']

### replace class characters with indices 
index = 0 
for x in classes:
   for i in range(len(labels)):
       if labels[i] == x:
           labels[i] = index
   index = index +1

labels = np.int8(labels)

### assemble x_train and y_train
y_train = labels
x_train = images.astype('float32')
x_train = x_train.reshape( (len(x_train),28,28,1))


### split data into class bins
data_0 = [ x for x,y in zip(x_train,y_train) if y == 0 ] 
data_1 = [ x for x,y in zip(x_train,y_train) if y == 1 ] 
data_2 = [ x for x,y in zip(x_train,y_train) if y == 2 ] 
data_3 = [ x for x,y in zip(x_train,y_train) if y == 3 ] 
data_4 = [ x for x,y in zip(x_train,y_train) if y == 4 ] 
data_5 = [ x for x,y in zip(x_train,y_train) if y == 5 ] 
data_6 = [ x for x,y in zip(x_train,y_train) if y == 6 ] 
data_7 = [ x for x,y in zip(x_train,y_train) if y == 7 ] 
data_8 = [ x for x,y in zip(x_train,y_train) if y == 8 ] 
data_9 = [ x for x,y in zip(x_train,y_train) if y == 9 ] 

np.random.shuffle(data_0)
np.random.shuffle(data_1)
np.random.shuffle(data_2)
np.random.shuffle(data_3)
np.random.shuffle(data_4)
np.random.shuffle(data_5)
np.random.shuffle(data_6)
np.random.shuffle(data_7)
np.random.shuffle(data_8)
np.random.shuffle(data_9)

### split each class bin into labelled, unlabelled and evaluation
tresholds_0 = [30000, 32000, 35000] #[3000, 32313, 39723]
tresholds_1 = [700, 7500, 8200] #[710, 7583, 8293]
tresholds_2 = [600, 6500, 7000] #[621, 6634, 7255]
tresholds_3 = [600, 6500, 7000] #[610, 6513, 7213]
tresholds_4 = [300, 3300, 3600] #[310, 3309, 3619]
tresholds_5 = [300, 2200, 2500] #[218, 2328, 2546]
tresholds_6 = [200, 700, 800] #[68, 734, 802]
tresholds_7 = [150, 200, 250] #[22, 238, 260]
tresholds_8 = [150, 200, 220] #[19, 210, 229]
tresholds_9 = [100, 120, 150] #[12, 138, 150]

# labelled set
x_train_shorten = data_0[0:tresholds_0[0]] + data_1[0:tresholds_1[0]] + data_2[0:tresholds_2[0]] + data_3[0:tresholds_3[0]] + data_4[0:tresholds_4[0]] + data_5[0:tresholds_5[0]] + data_6[0:tresholds_6[0]] + data_7[0:tresholds_7[0]] + data_8[0:tresholds_8[0]] + data_9[0:tresholds_9[0]]

y_train_shorten = list(np.zeros(tresholds_0[0])) + list(np.ones(tresholds_1[0])) + list( np.ones(tresholds_2[0]) * 2 ) + list( np.ones(tresholds_3[0]) * 3 ) + list( np.ones(tresholds_4[0]) * 4 ) + list( np.ones(tresholds_5[0]) * 5 ) + list( np.ones(tresholds_6[0]) * 6 ) + list( np.ones(tresholds_7[0]) * 7 ) + list( np.ones(tresholds_8[0]) * 8 ) + list( np.ones(tresholds_9[0]) * 9 )

x_train_shorten = np.array( x_train_shorten )
y_train_shorten = np.array( y_train_shorten )

#unlabelled set
x_train_unlabelled = data_0[tresholds_0[0]:tresholds_0[1]] + data_1[tresholds_1[0]:tresholds_1[1]] + data_2[tresholds_2[0]:tresholds_2[1]] + data_3[tresholds_3[0]:tresholds_3[1]] + data_4[tresholds_4[0]:tresholds_4[1]] + data_5[tresholds_5[0]:tresholds_5[1]] + data_6[tresholds_6[0]:tresholds_6[1]] + data_7[tresholds_7[0]:tresholds_7[1]] + data_8[tresholds_8[0]:tresholds_8[1]] + data_9[tresholds_9[0]:tresholds_9[1]]

y_train_unlabelled = list(np.zeros(tresholds_0[1] - tresholds_0[0])) + list(np.ones(tresholds_1[1] - tresholds_1[0])) + list( np.ones(tresholds_2[1] - tresholds_2[0]) * 2 ) + list( np.ones(tresholds_3[1] - tresholds_3[0]) * 3 ) + list( np.ones(tresholds_4[1] - tresholds_4[0]) * 4 ) + list( np.ones(tresholds_5[1] - tresholds_5[0]) * 5 ) + list( np.ones(tresholds_6[1] - tresholds_6[0]) * 6 ) + list( np.ones(tresholds_7[1] - tresholds_7[0]) * 7 ) + list( np.ones(tresholds_8[1] - tresholds_8[0]) * 8 ) + list( np.ones(tresholds_9[1] - tresholds_9[0]) * 9 )

x_train_unlabelled = np.array( x_train_unlabelled )
y_train_unlabelled = np.array( y_train_unlabelled )

#evaluation set
x_eval_shorten = data_0[tresholds_0[1]:tresholds_0[2]] + data_1[tresholds_1[1]:tresholds_1[2]] + data_2[tresholds_2[1]:tresholds_2[2]] + data_3[tresholds_3[1]:tresholds_3[2]] + data_4[tresholds_4[1]:tresholds_4[2]] + data_5[tresholds_5[1]:tresholds_5[2]] + data_6[tresholds_6[1]:tresholds_6[2]] + data_7[tresholds_7[1]:tresholds_7[2]] + data_8[tresholds_8[1]:tresholds_8[2]] + data_9[tresholds_9[1]:tresholds_9[2]]

#tresholds_3[1] -90
y_eval_shorten = list(np.zeros(tresholds_0[2] - tresholds_0[1])) + list(np.ones(tresholds_1[2] - tresholds_1[1])) + list( np.ones(tresholds_2[2] - tresholds_2[1]) * 2 ) + list( np.ones(tresholds_3[2] - tresholds_3[1]) * 3 ) + list( np.ones(tresholds_4[2] - tresholds_4[1]) * 4 ) + list( np.ones(tresholds_5[2] - tresholds_5[1]) * 5 ) + list( np.ones(tresholds_6[2] - tresholds_6[1]) * 6 ) + list( np.ones(tresholds_7[2] - tresholds_7[1]) * 7 ) + list( np.ones(tresholds_8[2] - tresholds_8[1]) * 8 ) + list( np.ones(tresholds_9[2] - tresholds_9[1]) * 9 )

x_eval_shorten = np.array( x_eval_shorten )
y_eval_shorten = np.array( y_eval_shorten )

### compile Classifier for pre-training
classifier.compile(optimizer='adam',             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),             metrics=['accuracy'])

### define pre-training and training sets
batch_size = 100

train_dataset = tf.data.Dataset.from_tensor_slices((x_train_shorten,y_train_shorten))
train_dataset = train_dataset.shuffle(1000).batch(batch_size)

pseudo_dataset = tf.data.Dataset.from_tensor_slices((x_train_unlabelled))
pseudo_dataset = pseudo_dataset.shuffle(1000).batch(batch_size)

eval_dataset = tf.data.Dataset.from_tensor_slices((x_eval_shorten,y_eval_shorten))
eval_dataset = eval_dataset.shuffle(1000).batch(batch_size)

### pre-train with labelled data
classifier.fit(train_dataset, validation_data=eval_dataset, batch_size=100, epochs=10)

### train with unlabelled data
avg_main_loss = tf.keras.metrics.Mean(name='avg_main_loss', dtype=tf.float32)
pseudo_steps = int(x_train_unlabelled.shape[0] / batch_size )
eval_steps = int(x_eval_shorten.shape[0] / batch_size )

epoch = 20

classifier_optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

x_values = np.array([])
threshold_pb = 0.75

for epoch_idx in range( epoch ):

  print( "Epoch {}/{}".format( epoch_idx + 1, epoch ) )
  classifier.evaluate(eval_dataset)

  for _ in range(5):

    pseudo_labels = classifier.predict( x_train_unlabelled ) 

    # only accept psuedo labels with certain threshold accuracy in their probability
    pseudo_labels = np.array( [ np.argmax(x) if np.max(x) > threshold_pb else -1 for x in pseudo_labels])
    dataset = np.array( [ [x,y] for x,y in zip( x_train_unlabelled, pseudo_labels ) if y != -1 ], dtype=object)
    x_values = np.array( [x for x,y in dataset ])
    y_values = np.array( [y for x,y in dataset ])

    pseudo_dataset = tf.data.Dataset.from_tensor_slices((x_values, y_values ))
    pseudo_dataset = pseudo_dataset.shuffle(1000).batch(batch_size)


    classifier.fit(pseudo_dataset, batch_size=50, epochs=1, verbose=0)
    classifier.fit(train_dataset, batch_size=50, epochs=1, verbose=0)

