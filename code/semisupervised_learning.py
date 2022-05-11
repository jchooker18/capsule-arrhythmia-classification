from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Softmax, Dropout, Conv1D, MaxPool1D, AveragePooling1D, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras import losses
from tensorflow.keras import Sequential
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from sklearn.utils import class_weight
import itertools
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.batch_size = 32
        self.num_classes = 5
        self.learnig_rate = 1e-3
        self.optimizer = tf.keras.optimizers.Adam(self.learnig_rate)
        self.epoch = 15
        

    def call(self, x_train, y_train, x_test, y_test, is_unsupervised=False):
        
        im_shape = (x_train.shape[1], 1)
    
        classifier = Sequential()
        
        classifier.add(Input(shape=(im_shape)))
        
        classifier.add(Conv1D(64, kernel_size=(6), activation='relu', input_shape=im_shape))
        classifier.add(BatchNormalization())
    
        classifier.add(MaxPool1D(pool_size=(3), strides=(2), padding="same"))
    
        classifier.add(Conv1D(64, kernel_size=(3), activation='relu'))
        classifier.add(BatchNormalization())
    
        classifier.add(MaxPool1D(pool_size=(2), strides=(2), padding="same"))
    
        classifier.add(Conv1D(64, kernel_size=(3), activation='relu'))
        classifier.add(BatchNormalization())
    
        classifier.add(MaxPool1D(pool_size=(2), strides=(2), padding="same"))
    
    # classifier.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    # classifier.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    # classifier.add(MaxPooling2D(pool_size=(2, 2)))
    
        classifier.add(Dropout(0.3))
        classifier.add(Flatten())
        classifier.add(Dense(64, activation='relu'))
        classifier.add(Dropout(0.15))
        classifier.add(Dense(32, activation='relu'))
        classifier.add(Dropout(0.15))
        classifier.add(Dense(self.num_classes, activation='softmax'))
    
        classifier.build()
        
        classifier.compile(optimizer='adam',loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

        if is_unsupervised ==False:
            history = classifier.fit(x_train,y_train, validation_data=(x_test,y_test), batch_size=self.batch_size, epochs=self.epoch)
        else:
            history = classifier.fit(x_train,y_train, validation_data=(x_test,y_test), batch_size=self.batch_size, epochs=1)
            
        return (classifier, history)
    
    def evaluate_model(self, history, x_test, y_test, classifier):
        scores = classifier.evaluate((x_test),y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))
        
        print(history)
        fig1, ax_acc = plt.subplots()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Classifier - Accuracy')
        plt.legend(['Training', 'Validation'], loc='lower right')
        plt.show()
        
        fig2, ax_loss = plt.subplots()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Classifier- Loss')
        plt.legend(['Training', 'Validation'], loc='upper right')
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.show()
        target_names=['0','1','2','3','4']
    
        y_true=[]
        for element in y_test:
            y_true.append(np.argmax(element))
        prediction_prob=classifier.predict(x_test)
        prediction=np.argmax(prediction_prob,axis=1)
        cnf_matrix = confusion_matrix(y_true, prediction)
        
        
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def add_noise(signal):
    noise = np.random.normal(0,0.3,186)
    return (signal+noise)

def dropout(signal):
    dropouts = np.random.randint(0,186,10)
    signal[dropouts] = 0
    return signal

### Import Data
train_df=pd.read_csv('data/mitbih_train.csv',header=None)
test_df=pd.read_csv('data/mitbih_test.csv',header=None)

### Data Distribution
train_df[187]=train_df[187].astype(int)
dist=train_df[187].value_counts()
print(dist)

plt.figure(figsize=(20,10))
my_circle=plt.Circle( (0,0), 0.7, color='white')
plt.pie(dist, labels=['n','q','v','s','f'], colors=['red','green','blue','skyblue','orange'],autopct='%1.1f%%')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()

### Upsample Data
df_1 = train_df[train_df[187]==1]
df_2 = train_df[train_df[187]==2]
df_3 = train_df[train_df[187]==3]
df_4 = train_df[train_df[187]==4]
df_0 = (train_df[train_df[187]==0]).sample(n=20000,random_state=42)

df_1_upsample = resample(df_1,replace=True,n_samples=20000,random_state=123)
df_2_upsample = resample(df_2,replace=True,n_samples=20000,random_state=124)
df_3_upsample = resample(df_3,replace=True,n_samples=20000,random_state=125)
df_4_upsample=resample(df_4,replace=True,n_samples=20000,random_state=126)

train_df = pd.concat([df_0, df_1_upsample, df_2_upsample, df_3_upsample, df_4_upsample])

### New distribution
dist=train_df[187].value_counts()
print(dist)

plt.figure(figsize=(20,10))
my_circle=plt.Circle( (0,0), 0.7, color='white')
plt.pie(dist, labels=['n','q','v','s','f'], colors=['red','green','blue','skyblue','orange'],autopct='%1.1f%%')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()

### Split into labelled and unlabelled data
ratio=0.05
df_0_lab = df_0.sample(frac=ratio, replace=False)
df_0_unl = df_0.drop(df_0_lab.index)
df_1_lab = df_1_upsample.sample(frac=ratio, replace=False)
df_1_unl = df_1_upsample.drop(df_1_lab.index)
df_2_lab = df_2_upsample.sample(frac=ratio, replace=False)
df_2_unl = df_2_upsample.drop(df_2_lab.index)
df_3_lab = df_3_upsample.sample(frac=ratio, replace=False)
df_3_unl = df_3_upsample.drop(df_3_lab.index)
df_4_lab = df_4_upsample.sample(frac=ratio, replace=False)
df_4_unl = df_4_upsample.drop(df_4_lab.index)


train_df_lab = pd.concat([df_0_lab, df_1_lab, df_2_lab, df_3_lab, df_4_lab])
train_df_unl = pd.concat([df_0_unl, df_1_unl, df_2_unl, df_3_unl, df_4_unl])

### Prepare datasets
y_train = train_df[187]
y_train_lab = train_df_lab[187]
y_train_unl = train_df_unl[187]
y_test = test_df[187]

y_train = to_categorical(y_train)
y_train_lab = to_categorical(y_train_lab)
y_train_unl = to_categorical(y_train_unl)
y_test = to_categorical(y_test)

x_train = train_df.iloc[:,:186].values
x_train_lab = train_df_lab.iloc[:,:186].values
x_train_unl = train_df_unl.iloc[:,:186].values
x_test = test_df.iloc[:,:186].values

for i in range(len(x_train_unl)):
    x_train_unl[i,:186]= add_noise(x_train_unl[i,:186])
    x_train_unl[i,:186]= dropout(x_train_unl[i,:186])

### Pre-training
model = Model()

x_train_lab = x_train_lab.reshape(len(x_train_lab), x_train_lab.shape[1],1)
x_train_unl = x_train_unl.reshape(len(x_train_unl), x_train_unl.shape[1],1)
x_test = x_test.reshape(len(x_test), x_test.shape[1],1)

classifier,history = model.call(x_train_lab, y_train_lab, x_test, y_test)
model.evaluate_model(history, x_test, y_test, classifier)
y_pred_lab = classifier.predict(x_test)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred_lab.argmax(axis=1))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cnf_matrix, classes=['N', 'S', 'V', 'F', 'Q'],normalize=True, title='Confusion matrix, with normalization')
plt.show()


thres = 0.82
unsup_epoch = 20
semi_epoch = 5
results=[]


opt = keras.optimizers.Adam(lr=4e-4)

classifier.compile(optimizer=opt,loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

for epoch_idx in range(unsup_epoch):
    print("Epoch {}/{}".format( epoch_idx + 1, unsup_epoch ))
    result = classifier.evaluate(x_test, y_test)
    result = np.array(result)
    results.append(result)
    
    for semi_idx in range(semi_epoch):
        print("SemiEpoch {}/{}".format( semi_idx + 1, semi_epoch ))
        y_pseudo = classifier.predict(x_train_unl)
        pseudo_labels = y_pseudo.argmax(axis=1)
        pseudo_prob = np.amax(y_pseudo, axis=1)
    
        # only accept psuedo labels with certain threshold accuracy in their probability
        accept_label = []
        for k in range(len(pseudo_labels)):
            #print(k)
            if pseudo_prob[k] > thres:
                accept_label.append(pseudo_labels[k])
            else:
                accept_label.append(-1)
        
        dataset = np.array([[x,y] for x,y in zip( x_train_unl, accept_label ) if y != -1 ], dtype=object)
        x_values = np.array( [x for x,y in dataset ])
        y_values = np.array( [y for x,y in dataset ])
        y_values.astype(float)
        
        classes = np.unique(y_values)
        if len(classes) > 1:
            h1,w1,n1 = x_values.shape
            x_values_reshaped = x_values.reshape((h1,w1))
            
            print(Counter(y_values))
            oversample = RandomOverSampler(sampling_strategy= 'auto')
            x_semi, y_semi = oversample.fit_resample(x_values_reshaped, y_values)
            x_semi, y_semi = resample(x_semi, y_semi,replace=True,n_samples=5000)
            h2,w2 = x_semi.shape
            x_semi = x_semi.reshape(h2,w2,1)
            print(Counter(y_semi))
            
            y_semi = to_categorical(y_semi, num_classes=5)
            
            classifier.fit(x_semi, y_semi, batch_size=32, epochs=1)
            classifier.fit(x_train_lab, y_train_lab, batch_size=32, epochs=1)

results = np.array(results)
fig3, ax_semi1 = plt.subplots()
l1, = plt.plot(results[:,1], color='red')
ax_semi2 = ax_semi1.twinx()
l2, = plt.plot(results[:,0], color ='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy & Loss')
plt.title('Semi-Supervised Training' )
plt.legend([l1,l2],['Loss', 'Accuracy'], loc='lower right')
plt.show()

# Compute confusion matrix
y_pred_final = classifier.predict(x_test)
cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred_final.argmax(axis=1))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cnf_matrix, classes=['N', 'S', 'V', 'F', 'Q'],normalize=True, title='Confusion matrix, with normalization')
plt.show()

