"""
2/25/17   Morgan Bryant

Arguments/Parameters: Invoke as 'python ModIAN.py -h' or 'ipython ModIAN.py -- -h'

See info.txt for information.  
This was tensorflow implementation was adapted from my mram.py, an mult-attention
extension of recurrent-attention-model.

This is a basic implementation of a psychologically-grounded neural network model 
that uses a simple selection scheme that extends the model's ability to perform
significant transferrence of aggregated skill.

[ see associated documents for details. ]

"""

#########################   IMPORTS & VERSIONS & SETUP #########################
import tensorflow as tf
import tf_mnist_loader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time, random, sys, os, argparse
from datetime import datetime
startTime = datetime.now()

dataset = tf_mnist_loader.read_data_sets("mnist_data", one_hot=True)

inp_sz = 5;     lr = 0.001
N = 10;         epochs = 50
theta = 8;      batchsize = 16
B = 4;          disp_step = 10
phi = 16;       momentum = 0.9
out_sz = 2

#inp_sz = 784;       lr = 0.001
#N = 32;             epochs = 50
#theta = 32;         batchsize = 16
#B = 32;             disp_step = 10
#phi = 32;           momentum = 0.9
#out_sz = 10
#
# DATA
#train_I, train_O = simple_task(200)
#test_I, test_O = simple_task(30)
def gate_variable(shape, myname, train):
    initial = tf.zeros(shape, dtype='float32')
    return tf.Variable(initial, name=myname, trainable=train)

def weight_variable(shape, myname, train):
    initial = tf.random_uniform(shape, minval = -0.1, maxval = 0.1, dtype='float32')
    return tf.Variable(initial, name=myname, trainable=train)

def model(inp, w):
  with tf.variable_scope("network", reuse=None):

    inp_r = tf.reshape(inp, (inp_sz, 1))
    W_I_XB_r = tf.reshape(w['W_I_XB'], (N*B, inp_sz))
    W_I_Xth_r = tf.reshape(w['W_I_Xth'], (N*theta, inp_sz))
    W_Y_X_new = tf.reshape(w['W_Y_X1'], (out_sz, N*phi))
    X_B  = tf.matmul(W_I_XB_r, inp_r )
    X_B = tf.nn.softmax(X_B)
    X_th = tf.matmul(W_I_Xth_r, inp_r )

    D = 'float32'
    x_th = tf.reshape(X_th, (N, theta, 1))
    x_b = tf.reshape(X_B, (N, B, 1))

    M_th_B = tf.reshape(w['M'], (B * phi, theta))
    def module_operation_TH(x_th):
        return tf.matmul(M_th_B, x_th) # shape: (B*phi, N). -> (N*B, phi)
    def module_operation_B(i):
        Z_slice = tf.reshape(tf.slice(Zr, [i,0,0],[1,-1,-1]), (phi,B))
        xb_slice = tf.reshape(tf.slice(x_b, [i,0,0],[1,-1,-1]), (B,1))
        return tf.matmul(Z_slice, xb_slice)

    Z = tf.map_fn(module_operation_TH, x_th, dtype=D)
    Zr = tf.reshape(Z, (N,  phi, B,))
    Y = tf.map_fn(module_operation_B, tf.range(N), dtype=D) # map_fn: for parallelization

    Yr = tf.reshape(Y, (N*phi,1))
    FC_Y_Xnew = tf.reshape(w['W_Y_X1'], (out_sz, N*phi))
    X_new = tf.nn.relu( tf.matmul(W_Y_X_new, Yr) )
    return tf.reshape(X_new, (1,out_sz))

# I -> X_t.  I -> X_B.  sigma( (MX_t)sigma(X_B) ) -> Y.  Y -> X'.
#   inp_sz = 5;     lr = 0.001
#   N = 10;         epochs = 10
#   theta = 8;      batchsize = 16
#   B = 4;          disp_step = 10
#   phi = 16        momentum = 0.9
#   out_sz = 2

Weights_0 = {
  'W_I_Xth': weight_variable((inp_sz, N, theta), "Weights_Inp_to_Xtheta",True),
  #'b_I_Xth': weight_variable((1, N, theta), "biases_Inp_to_Xtheta", True),
  'W_I_XB' : gate_variable((inp_sz, N, B), "Weights_Inp_to_XB", True),
  #'b_I_XB' : weight_variable((1, N, B), "biases_Inp_to_XB", True),
  'M'      : weight_variable( (B, theta, phi), "Module", True), # trainable
  'W_Y_X1' : weight_variable( (N, phi, out_sz), "Weights_Y_to_Xprime", True),
  #'b_Y_X1' : weight_variable( (N, 1, out_sz), "biases_Y_to_Xprime", True),
}

x_var = tf.placeholder("float32", [1,inp_sz])
y_var = tf.placeholder("float32", [1,out_sz])
this_model = model(x_var, Weights_0)
print "SHAPE:",this_model.get_shape(), y_var.get_shape()

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits\
        (logits = this_model, labels = tf.to_float(y_var)))
#optimizer = tf.train.MomentumOptimizer(learning_rate = lr, \
#        momentum = momentum, ).minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr)
grads = optimizer.compute_gradients(y_var)
grad_var = tf.gradients(cost, Weights_0.values())[0]
optimizer = optimizer.minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        #avg_cost = 0.0
        for i in range(batchsize):
            R = np.random.randint(200)
            #x, y = train_I[R,:], np.expand_dims(train_O[R,:], 1)
            x,y = dataset.train.next_batch(1)
            x = np.expand_dims(x[0,:],0)

            ValsOfInterest = [optimizer, cost, grad_var]#+Weights_0.values()
            #for v in ValsOfInterest: print '\t',v
            c = sess.run(ValsOfInterest, feed_dict={x_var: x, y_var: y})

#            c = sess.run([optimizer, cost], feed_dict={x_var: x, y_var: y})
        print "Epoch", epoch, "Cost", c
        if epoch % disp_step==0:
            pass #print "Epoch", epoch, "Cost", c, 
    print "Opt finished."

    corr_pred = tf.equal(tf.argmax(this_model, 1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(corr_pred, "float"))
    x,y = np.squeeze(dataset.test.next_batch(1))
    print "Accuracy:", np.mean( [accuracy.eval({x_var: x, y_var: y})
                for r in range(30)])

print "Done."
