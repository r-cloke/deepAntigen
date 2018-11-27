# -*- coding: utf-8 -*-
"""
@author: Ryan Cloke
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib
import random
import scipy.stats
import os

path = os.getcwd()

def binaryEncode(peptide):
    """
    A	Alanine
    R	Arginine
    N	Asparagine
    D	Aspartic acid
    C	Cysteine
    Q	Glutamine
    E	Glutamic acid
    G	Glycine
    H	Histidine
    I	Isoleucine
    L	Leucine
    K	Lysine
    M	Methionine
    F	Phenylalanine
    P	Proline
    S	Serine
    T	Threonine
    W	Tryptophan
    Y	Tyrosine
    V	Valine
    """

    #do 1 hot encoding
    binaryPeptide=''
    for aa in peptide:
        if aa =='A':
            binaryAmino='10000000000000000000'
        if aa =='R':
            binaryAmino='01000000000000000000'
        if aa =='N':
            binaryAmino='00100000000000000000'
        if aa =='D':
            binaryAmino='00010000000000000000'
        if aa =='C':
            binaryAmino='00001000000000000000'
        if aa =='Q':
            binaryAmino='00000100000000000000'
        if aa =='E':
            binaryAmino='00000010000000000000'
        if aa =='G':
            binaryAmino='00000001000000000000'
        if aa =='H':
            binaryAmino='00000000100000000000'
        if aa =='I':
            binaryAmino='00000000010000000000'
        if aa =='L':
            binaryAmino='00000000001000000000'
        if aa =='K':
            binaryAmino='00000000000100000000'
        if aa =='M':
            binaryAmino='00000000000010000000'
        if aa =='F':
            binaryAmino='00000000000001000000'
        if aa =='P':
            binaryAmino='00000000000000100000'
        if aa =='S':
            binaryAmino='00000000000000010000'
        if aa =='T':
            binaryAmino='00000000000000001000'
        if aa =='W':
            binaryAmino='00000000000000000100'
        if aa =='Y':
            binaryAmino='00000000000000000010'
        if aa =='V':
            binaryAmino='00000000000000000001'
        binaryPeptide=binaryPeptide +binaryAmino
        
    if len(binaryPeptide) == (20*9):
        binaryPeptide = np.array(list(binaryPeptide),dtype=float)
        binaryPeptide = np.reshape(binaryPeptide,(binaryPeptide.shape[0],1))
        binaryPeptide = np.transpose(binaryPeptide)
        return binaryPeptide

   
def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32,name="X",shape=(n_x,180))
    Y = tf.placeholder(tf.float32,name="Y",shape=(n_y,1))
    return X, Y

def initialize_parameters():   
    tf.set_random_seed(1)                  
        
    W1 = tf.get_variable("W1", [100,180], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [100,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [1,100], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [1,1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    Z1 = tf.add(tf.matmul(W1,tf.transpose(X)),b1)                                
    A1 = tf.nn.relu(Z1)                                              
                   
    Z2 = tf.add(tf.matmul(W2,A1),b2)                                              
   
    return Z2

def compute_cost(Z2, Y,parameters):
    logits = tf.transpose(Z2)
    beta=0.1
    regularizer = tf.nn.l2_loss(parameters['W2'])
    cost = tf.losses.mean_squared_error(labels=logits,predictions=Y) + (beta*regularizer)
    return cost

def model(X_train, Y_train, X_test,learning_rate,
          num_epochs, print_cost = True):
   
    ops.reset_default_graph()                         
    tf.set_random_seed(1)                            
    (n_x, m) = X_train.shape                         
    n_y = Y_train.shape[0]                            
    
    X, Y = create_placeholders(n_x, n_y)

    parameters = initialize_parameters()
    

    Z2 = forward_propagation(X, parameters)

    cost = compute_cost(Z2, Y, parameters)

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    costLst=[]
    epochLst=[]
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            aoptimizer,acost = sess.run([optimizer,cost], feed_dict={X: X_train, Y: Y_train})
            if epoch%25 == 0:
                costLst.append(acost)
                epochLst.append(epoch)
  
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")  
        saver = tf.train.Saver()
        saver.save(sess, path+'\\my_test_model',global_step=500)
        
        Z2 = tf.transpose(tf.reshape(Z2,[1,X_train.shape[0]]))
        correct_prediction = tf.equal(tf.to_int32(Z2>0.95),tf.to_int32(Y>0.95))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        
        print('training pearson correlation coef is: ',scipy.stats.pearsonr(Z2.eval({X: X_train}),Y.eval({Y: Y_train}))[0])
        test_X = tf.placeholder(tf.float32,name="test_X",shape=(X_test.shape[0],180))
        test_Y = tf.placeholder(tf.float32,name="test_Y",shape=(Y_test.shape[0],1))
        init = tf.global_variables_initializer()
        pred = forward_propagation(test_X,parameters)
        
        output = sess.run(pred,feed_dict={test_X:X_test})

        savePred = output.tolist()
        output = tf.transpose(tf.reshape(output,[1,X_test.shape[0]]))
        correct_prediction = tf.equal(tf.to_int32(output>0.95),tf.to_int32(test_Y>0.95))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print ("Test Accuracy:", accuracy.eval({test_X: X_test, test_Y: Y_test}))
        print('test pearson correlation coef is: ',scipy.stats.pearsonr(output.eval({test_X: X_test}),test_Y.eval({test_Y: Y_test}))[0])

        with open('predictions.csv','w') as predFile:
            predFile.write('prediction'+','+'assay value'+'\n')
            for i in range(len(savePred[0])):
                predFile.write(str(savePred[0][i])+','+str(Y_test.tolist()[i]).strip('[').strip(']')+'\n')
        
        predFile.close()
        sess.close()
        matplotlib.pyplot.plot(epochLst,costLst)
        
        return parameters        

def getXData():
    pepLst=[]
    responseLst=[]
    
    #randomize input file order
    lines = open(path+'\\9mer_HLA_A_0201.txt').readlines()[1:]
    random.shuffle(lines)

    for i in range(len(lines)):
        seq = lines[i].split('\t')[3].strip('\n')
        binaryPeptide = binaryEncode(seq)
        pepLst.append(binaryPeptide)
            
        response = lines[i].split('\t')[5].strip('\n')
        response = 1-(np.log10(float(response)) / np.log10(14000000))
        responseLst.append(response)
            
    X = np.array(pepLst) 
    print(X.shape)
    X = X[:,0, :]
    
    Y =np.transpose(np.array(responseLst))
    Y = np.reshape(Y,(Y.shape[0],1))
    
    return X,Y

(Xdataset,Ydataset) = getXData()

X_train = Xdataset[0:4832,0:181]
X_test = Xdataset[4832:6039,0:181]

Y_train = Ydataset[0:4832]
Y_test = Ydataset[4832:6039] 
Y_train = np.reshape(Y_train,(Y_train.shape[0],1))
Y_test = np.reshape(Y_test,(Y_test.shape[0],1))

parameters = model(X_train, Y_train, X_test,learning_rate = 0.001,num_epochs = 500, print_cost = True)