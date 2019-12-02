# TextCNN model

# python-3.6.2
# tensorflow-1.2.1
# keras-2.0.7
# Author : Paul
# All right reserved

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import time
from keras import backend as K
import string
from string import digits
import numpy as np
from tensorflow.contrib import learn
# import pickle

K.set_image_dim_ordering('th')
# with open('C:/Users/admin/Desktop/input_x_y.pkl', 'rb') as g:
#     input_mat_train, input_mat_test, y_train, y_test = pickle.load(g)

# text preprocessing

with open('C:/Users/admin/Desktop/rt-polarity.neg.txt', 'r', encoding='utf-8') as f:
    file = f.read()
    neg = file.split(' . \n')
neg1 = []
for mail in neg:
    trantab = str.maketrans({key: None for key in string.punctuation})
    j = mail.translate(trantab).replace('\n', ' ').replace('—', ' ').replace(' i e ', ' ').replace('…', ' ').replace(
        '–', ' ').replace('"', ' ').replace('[', ' ').replace(']', ' ')
    remove_digits = str.maketrans('', '', digits)
    k = j.translate(remove_digits)
    l = k.replace('   ', ' ').replace('  ', ' ')
    neg1.append(l)
neg1 = neg1[:-1]
# print(neg1)
neg1 = neg1[:1000]
with open('C:/Users/admin/Desktop/rt-polarity.pos.txt', 'r', encoding='utf-8') as f:
    file = f.read()
    pos = file.split(' . \n')
pos1 = []
for mail in pos:
    trantab = str.maketrans({key: None for key in string.punctuation})
    j = mail.translate(trantab).replace('\n', ' ').replace('—', ' ').replace(' i e ', ' ').replace('…', ' ').replace(
        '–', ' ').replace('"', ' ').replace('[', ' ').replace(']', ' ')
    remove_digits = str.maketrans('', '', digits)
    k = j.translate(remove_digits)
    l = k.replace('   ', ' ').replace('  ', ' ')
    pos1.append(l)
pos1 = pos1[:-1]
# print(pos1)
pos1 = pos1[:1000]

total = pos1 + neg1
# print(len(total))

# max length of mails
max_length = max([len(x.split(' ')) for x in total])
print(max_length)  # 94

# Y = one-hot label 1, 0
negative_label = [[1, 0] for _ in neg1]
positive_label = [[1, 0] for _ in pos1]
y = np.concatenate([positive_label, negative_label], axis=0)

# vector of index of words in mail
vocab_processor = learn.preprocessing.VocabularyProcessor(max_length)
x = np.array(list(vocab_processor.fit_transform(total)))
# print(x[:3])
# [[ 1  2  3  4  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
# ...

# shuffle the data set
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
# divide the data into training set and testing set
dev_sample_index = -1 * int(0.2 * float(len(y)))
x_train, x_test = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_test = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

# word-to-vector matrix
Matrix = np.random.randn(len(vocab_processor.vocabulary_) * 10).reshape(len(vocab_processor.vocabulary_), 10)
print(Matrix)
print(Matrix.shape)

# transform x into matrix
input_mat_train = np.random.rand(10)
for i in range(0, x_train.shape[0]):
    vec = x_train[i]
    for index in range(0, max_length):
        input_mat_train = np.vstack((input_mat_train, Matrix[vec[index], :]))
    print(i)
input_mat_train = input_mat_train[1:, ]
print(input_mat_train.shape)

input_mat_test = np.random.rand(10)
for i in range(0, x_test.shape[0]):
    vec = x_test[i]
    for index in range(0, max_length):
        input_mat_test = np.vstack((input_mat_test, Matrix[vec[index], :]))
    print(i)
input_mat_test = input_mat_test[1:, ]
print(input_mat_test.shape)

# reshape the data (num,pixel,row,column)
print(y_train.shape)
print(input_mat_train.shape)
X_train = input_mat_train.reshape(1600, 1, 74, 10).astype('float32')
X_test = input_mat_test.reshape(400, 1, 74, 10).astype('float32')
print(X_train[0])
seed = 7
np.random.seed(seed)

# construct CNN model
def cnn_model():
    # create model
    model = Sequential()
    model.add(Conv2D(50, (2, 10), input_shape=(1,74, 10), activation='relu'))
    model.add(MaxPooling2D(pool_size=(73, 1)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(30, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model
# print(K.image_data_format())

# CNN
t1 = time.time()
# build the model
cmod = cnn_model()
# Fit the model
hist = cmod.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=50, epochs=30, verbose=2)
# Final evaluation of the model
scores = cmod.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
t2 = time.time()
print("time: ", t2 - t1)
