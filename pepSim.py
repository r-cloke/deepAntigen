# -*- coding: utf-8 -*-
"""
@author: Ryan Cloke
"""

import random
import numpy as np
import tensorflow as tf
import os

path = os.getcwd()

def getRandomAA():
    aminoAcid = ''
    for aa in range(9):
        index = random.randint(0,20)
    
        binaryString=''
        for i in range(20):
            if i == index:
                aaBinary='1'
            else:
                aaBinary='0'
            binaryString = binaryString+ aaBinary
            
        aminoAcid = aminoAcid + binaryString
    
    aminoAcid = np.array(list(aminoAcid),dtype=float)
    aminoAcid = np.reshape(aminoAcid,(aminoAcid.shape[0],1))
    aminoAcid = np.transpose(aminoAcid)
    return aminoAcid

def simulatePeptides(num):
    pepLst=[]
    for peptide in range(num):
        randoPeptide = getRandomAA()
        pepLst.append(randoPeptide)
        
    X = np.array(pepLst).astype(np.float32)
    X = np.reshape(X,(num,180))
    
    return X

def binaryToAA(binary):
    decoded=''
    for i in range(int(len(binary)/9)):
        aa=binary[20*i:(20*i+20)]
        aa=''.join(str(int(e)) for e in aa)
        binaryAmino=''
        if aa =='10000000000000000000':
            binaryAmino='A'
        if aa =='01000000000000000000':
            binaryAmino='R'
        if aa =='00100000000000000000':
            binaryAmino='N'
        if aa =='00010000000000000000':
            binaryAmino='D'
        if aa =='00001000000000000000':
            binaryAmino='C'
        if aa =='00000100000000000000':
            binaryAmino='Q'
        if aa =='00000010000000000000':
            binaryAmino='E'
        if aa =='00000001000000000000':
            binaryAmino='G'
        if aa =='00000000100000000000':
            binaryAmino='H'
        if aa =='00000000010000000000':
            binaryAmino='I'
        if aa =='00000000001000000000':
            binaryAmino='L'
        if aa =='00000000000100000000':
            binaryAmino='K'
        if aa =='00000000000010000000':
            binaryAmino='M'
        if aa =='00000000000001000000':
            binaryAmino='F'
        if aa =='00000000000000100000':
            binaryAmino='P'
        if aa =='00000000000000010000':
            binaryAmino='S'
        if aa =='00000000000000001000':
            binaryAmino='T'
        if aa =='00000000000000000100':
            binaryAmino='W'
        if aa =='00000000000000000010':
            binaryAmino='Y'
        if aa =='00000000000000000001':
            binaryAmino='V'
        decoded=decoded+binaryAmino
    return decoded
    
X = simulatePeptides(100000)

with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph(path + '\\my_test_model-500.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    W1 = sess.run('W1:0')
    b1 = sess.run('b1:0')
    W2 = sess.run('W2:0')
    b2 = sess.run('b2:0')
    
    Z1 = np.add(np.dot(W1,np.transpose(X)),b1)
    A1 = np.maximum(Z1,0)
    Z2 = np.add(np.dot(W2,A1),b2)
    
    bindingMax = Z2.max()
    maxPos = np.argmax(Z2)
    
    bestBinary = X[maxPos][0:181]
    print(binaryToAA(bestBinary),' had the top binding score of ',bindingMax)
