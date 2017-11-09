# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 20:33:17 2017

@author: lenovo
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
import os

INPUT_SIGNAL_TYPES = [ "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"]

LABELS=["WALKING", 
    "WALKING_UPSTAIRS", 
    "WALKING_DOWNSTAIRS", 
    "SITTING", 
    "STANDING", 
    "LAYING"]

DATA_PATH = "D:/UCI HAR Dataset/UCI HAR Dataset/"

def load_X(X_signals_paths):
   X_signals=[]
   for signal_type_path in X_signals_paths:
       file = open(signal_type_path,'r')
       X_signals.append([np.array(series,dtype=np.float32) for series in [row.replace('  ',' ').strip().split(' ') for row in file]])
       file.close()
   return np.transpose(np.array(X_signals),(1,2,0))

X_train_signals_path = [DATA_PATH+"train/Inertial Signals/"+signal+"train.txt" for signal in INPUT_SIGNAL_TYPES]
X_test_signals_path = [DATA_PATH+"test/Inertial Signals/"+signal+"test.txt" for signal in INPUT_SIGNAL_TYPES]

X_train = load_X(X_train_signals_path)
X_test = load_X(X_test_signals_path)


def load_Y(Y_signal_path):
    file = open(Y_signal_path)
    y_ = np.array([elem for elem in [row.replace('  ',' ').strip().split(' ') for row in file]], dtype=np.int32)  
    file.close()
    return y_ - 1

y_train_path = DATA_PATH+"train/"+"y_train.txt"
y_test_path = DATA_PATH+"test/"+"y_test.txt"

Y_train = load_Y(y_train_path)

Y_test = load_Y(y_test_path)

training_data_count = len(X_train)
test_data_count = len(X_test)  # 2947 testing series
n_steps = len(X_train[0])  # 128 timesteps per series
n_input = len(X_train[0][0])


n_hidden = 32
n_classes = 6

learning_rate = 0.0025
lambda_loss_amount = 0.0015
training_iters = training_data_count*300
batch_size = 1500
display_iter = 30000

#def LSTM_RNN(_X, _weights,_biases):
#    _X = tf.transpose(_X,[1,0,2])
#    _X = tf.reshape(_X,[-1,n_input])
#    
#    _X = tf.nn.relu(tf.matmul(_X,_weights['hidden'])+_biases['hidden'])
#    _X = tf.split(_X,n_steps,0)
#    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1.0,state_is_tuple=True)
#    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1.0,state_is_tuple=True)
#    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1,lstm_cell_2], state_is_tuple=True)
#    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells,_X,dtype=tf.float32)
#    lstm_last_output = outputs[-1]
#    
#    return tf.matmul(lstm_last_output,_weights['out'])+_biases['out']
#
#
#def extract_batch_size(_train,step,batch_size):
#    shape = list(_train.shape)
#    shape[0]=batch_size
#    batch_s = np.empty(shape)
#    
#    for i in range(batch_size):
#        index = ((step-1)*batch_size)%len(_train)
#        batch_s[i]=_train[index]
#    return batch_s
#
#def one_hot(y_):
#    y_ = y_.reshape(len(y_))
#    n_values = int(np.max(y_))+1
#    return np.eye(n_values)[np.array(y_, dtype=np.int32)]
#

def LSTM_RNN(_X, _weights, _biases):
    # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters. 
    # Moreover, two LSTM cells are stacked which adds deepness to the neural network. 
    # Note, some code of this notebook is inspired from an slightly different 
    # RNN architecture used on another dataset, some of the credits goes to 
    # "aymericdamien" under the MIT license.

    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) 
    # new shape: (n_steps*batch_size, n_input)
    
    # Linear activation
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0) 
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # Get last time step's output feature for a "many to one" style classifier, 
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]
    
    # Linear activation
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


def extract_batch_size(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data. 
    
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index] 

    return batch_s


def one_hot(y_):
    # Function to encode output labels from number indexes 
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

x = tf.placeholder(tf.float32,[None,n_steps,n_input])
y=tf.placeholder(tf.float32,[None,n_classes])

weights={
        'hidden':tf.Variable(tf.random_normal([n_input,n_hidden])),
        'out':tf.Variable(tf.random_normal([n_hidden,n_classes], mean=1.0))
        }

biases={
        'hidden':tf.Variable(tf.random_normal([n_hidden])),
        'out':tf.Variable(tf.random_normal([n_classes]))
        }

pred = LSTM_RNN(x,weights,biases)

l2 = lambda_loss_amount*sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))+l2

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

test_losses = []
train_losses=[]
test_accuracies=[]
train_accuracies=[]

sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()
sess.run(init)

step = 1
while step * batch_size <= training_iters:
    batch_xs =         extract_batch_size(X_train, step, batch_size)
    batch_ys = one_hot(extract_batch_size(Y_train, step, batch_size))

    # Fit training using batch data
    _, loss, acc = sess.run(
        [optimizer, cost, accuracy],
        feed_dict={
            x: batch_xs, 
            y: batch_ys
        }
    )
    train_losses.append(loss)
    train_accuracies.append(acc)
    
    # Evaluate network only at some steps for faster training: 
    if (step*batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):
        
        # To not spam console, show training accuracy/loss in this "if"
        print("Training iter #" + str(step*batch_size) + \
              ":   Batch Loss = " + "{:.6f}".format(loss) + \
              ", Accuracy = {}".format(acc))
        
        # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
        loss, acc = sess.run(
            [cost, accuracy], 
            feed_dict={
                x: X_test,
                y: one_hot(Y_test)
            }
        )
        test_losses.append(loss)
        test_accuracies.append(acc)
        print("PERFORMANCE ON TEST SET: " + \
              "Batch Loss = {}".format(loss) + \
              ", Accuracy = {}".format(acc))

    step += 1

print("Optimization Finished!")

#one_hot_predictions, 
