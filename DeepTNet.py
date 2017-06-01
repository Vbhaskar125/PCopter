import tensorflow as tf
import pandas as pd
import numpy as np

x = tf.placeholder("float", [None, 9])
y = tf.placeholder("float", [None, 3])

class DeepTNet:
    def __init__(self,n_input=9, n_layer1=6,n_layer2=4, n_output=3,learningRate=0.001,epoch = 20,batchSize = 10):
        self.learningRate=learningRate
        self.n_input=n_input
        self.n_layer1=n_layer1
        self.n_layer2=n_layer2
        self.n_output=n_output
        self.epoch =epoch
        self.batchSize=batchSize
        x = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("float", [None, n_output])
        self.weights = {
            'w1': tf.Variable(tf.random_normal([n_input, n_layer1])),
            'w2': tf.Variable(tf.random_normal([n_layer1, n_layer2])),
            'out': tf.Variable(tf.random_normal([n_layer2, n_output]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([n_layer1])),
            'b2': tf.Variable(tf.random_normal([n_layer2])),
            'out': tf.Variable(tf.random_normal([n_output]))
        }
        self.saver = tf.train.Saver()


    def MLP(self,x):
        layer1 = tf.add(tf.matmul(x, self.weights['w1']), self.biases['b1'])
        layer1 = tf.nn.tanh(layer1)
        layer2 = tf.add(tf.matmul(layer1, self.weights['w2']), self.biases['b2'])
        layer2 = tf.nn.sigmoid(layer2)
        outlayer = tf.add(tf.matmul(layer2, self.weights['out']), self.biases['out'])
        outlayer = tf.nn.sigmoid(outlayer)
        return outlayer

    def prepData(self,data):
        a=[]
        length= len(data)
        for qw in xrange(0,length):
            a.append(data[qw])
        yvar=[]
        for qw in xrange(len(a)):
            yvar.append(a[qw][8])
        xvar=[]
        for qw in xrange(0,len(a)):
            xvar.append([a[qw].player_vel,a[qw].player_dist_to_ceil,a[qw].next_gate_block_top,a[qw].next_gate_dist_to_player,a[qw].next_gate_block_bottom,a[qw].player_dist_to_floor,a[qw].player_y,a[qw].lives,a[qw].action])

        return xvar,yvar


    def convertToOneHot(self,vector,num_classes):
        result = np.zeros(shape=(len(vector), num_classes))
        result[np.arange(len(vector)), vector] = 1
        return result.astype(int)



    def train(self,inpt):
        inpX,inpY=self.prepData(inpt)
        inpY=self.convertToOneHot(inpY,3)
        pred = self.MLP(x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        train = tf.train.GradientDescentOptimizer(self.learningRate).minimize(cost)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(self.epoch):
                totalBatch = len(inpY)/self.batchSize
                for z in range(totalBatch):
                    x_batch = inpX[z * self.batchSize:(z + 1) * self.batchSize]
                    y_batch = inpY[z * self.batchSize:(z + 1) * self.batchSize]
                    #y_batch=np.reshape(y_batch,(None,2))
                    c = sess.run([train, cost], feed_dict={x: x_batch, y: y_batch})
            self.saver.save(sess, "/home/ecotine/Desktop/Shots/model.ckpt")


    def predict(self,gamestate,lives):
        inpredict=self.getPredictDataO(gamestate,lives)
        #inpredict.append(0.0)
        pred=self.MLP(x)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            c=sess.run([pred],feed_dict={x:inpredict})
        print c
        oa=np.argmax(c)
        inpredict = self.getPredictDataA(gamestate, lives)
        # inpredict.append(0.0)
        pred = self.MLP(x)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            c = sess.run([pred], feed_dict={x: inpredict})
        ob=np.argmax(c)
        return max(oa,ob)

    def getPredictDataO(self,gamestate,lives):
        xvar=[]
        xvar=([gamestate['player_vel'],gamestate['player_dist_to_ceil'],gamestate['next_gate_block_top'],gamestate['next_gate_dist_to_player'],gamestate['next_gate_block_bottom'],gamestate['player_dist_to_floor'],gamestate['player_y'],lives,1])
        #xvar.append([gamestate['player_vel'], gamestate['player_dist_to_ceil'], gamestate['next_gate_block_top'], gamestate['next_gate_dist_to_player'], gamestate['next_gate_block_bottom'], gamestate['player_dist_to_floor'], gamestate['player_y'], lives,0])
        print("-----------------------------------------------")
        #xvar.append([0.])
        print("----------------------------")
        return xvar

    def getPredictDataA(self,gamestate,lives):
        xvar=[]
        xvar=([gamestate['player_vel'],gamestate['player_dist_to_ceil'],gamestate['next_gate_block_top'],gamestate['next_gate_dist_to_player'],gamestate['next_gate_block_bottom'],gamestate['player_dist_to_floor'],gamestate['player_y'],lives,0])
        #xvar.append([gamestate['player_vel'], gamestate['player_dist_to_ceil'], gamestate['next_gate_block_top'], gamestate['next_gate_dist_to_player'], gamestate['next_gate_block_bottom'], gamestate['player_dist_to_floor'], gamestate['player_y'], lives,0])
        print("-----------------------------------------------")
        #xvar.append([0.])
        print("----------------------------")
        return xvar

