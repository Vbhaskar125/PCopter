#working
from __future__ import print_function
import tensorflow as tf
import pandas as pd
import numpy as np
data=pd.read_csv("/home/ecotine/Desktop/presentDS/BreastCancer.csv")
data["diagnosis"]= data["diagnosis"].map({'M':1,'B':0})
data.drop('id',axis=1,inplace=True)
data=data.reindex(np.random.permutation(data.index))
data=data.reindex(np.random.permutation(data.index))
data=data.reindex(np.random.permutation(data.index))
testD=int(data.shape[0]*0.3)
testData=data[:testD]
trainingData=data[testD:]
trainingoutput=trainingData["diagnosis"]
testoutput=testData["diagnosis"]
trainingData=trainingData.drop('diagnosis',axis=1,inplace=False)
testData=testData.drop('diagnosis',axis=1,inplace=False)
#parameters
learningRate=0.001
epoch=20
batchSize=10
display_step = 1
#networkConfig
n_input=30
n_layer1=20
n_layer2=10
n_layer3=8
n_output=1

#io placeholders
x=tf.placeholder("float",[None,n_input])
y=tf.placeholder("float",[None,n_output])


def multilayer_perceptron(x,weights,biases):
    layer1=tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    layer1=tf.nn.tanh(layer1)
    layer2=tf.add(tf.matmul(layer1, weights['w2']), biases['b2'])
    layer2=tf.nn.tanh(layer2)
    layer3=tf.add(tf.matmul(layer2, weights['w3']), biases['b3'])
    layer3=tf.nn.sigmoid(layer3)
    outlayer=tf.add(tf.matmul(layer3,weights['out']),biases['out'])
    outlayer=tf.nn.sigmoid(outlayer)
    return outlayer

#weights
weights={
    'w1':tf.Variable(tf.random_normal([n_input,n_layer1])),
    'w2':tf.Variable(tf.random_normal([n_layer1,n_layer2])),
    'w3':tf.Variable(tf.random_normal([n_layer2,n_layer3])),
    'out':tf.Variable(tf.random_normal([n_layer3,n_output]))
}
biases={
    'b1':tf.Variable(tf.random_normal([n_layer1])),
    'b2':tf.Variable(tf.random_normal([n_layer2])),
    'b3':tf.Variable(tf.random_normal([n_layer3])),
    'out':tf.Variable(tf.random_normal([n_output]))
}


#construct model
pred=multilayer_perceptron(x,weights,biases)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
train=tf.train.GradientDescentOptimizer(learningRate).minimize(cost)
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(epoch):
        totalBatch=40
        for z in range(totalBatch):
            x_batch=trainingData[z*batchSize:(z+1)*batchSize]
            y_batch=trainingoutput[z*batchSize:(z+1)*batchSize]
            temp=y_batch.shape
            y_batch=y_batch.values.reshape(temp[0],1)
            c=sess.run([train,cost], feed_dict={x:x_batch,y:y_batch})
            if(z%10==0):
                print(weights['w2'].eval())
            print("batch")
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(testData.shape)
    print(testoutput.shape)
    Yte = np.reshape(testoutput, (testoutput.shape[0], 1))
    print("Accuracy:", accuracy.eval({x: testData, y: Yte}))
    asz = sess.run(pred, feed_dict={x: testData})
    print(asz)
    print("-==-=-=--=-=-==-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-==================================")
    print(testoutput)

