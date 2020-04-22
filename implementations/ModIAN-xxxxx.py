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
import time, random, sys, os, argparse, time
from datetime import datetime
startTime = datetime.now()

dataset = tf_mnist_loader.read_data_sets("mnist_data", one_hot=True)

inp_sz = 8;     lr = 0.00005 ## super small learning rate bc multiplicative effects
N = 8;          epochs = 40000
theta = 8;      batchsize = 16
B = 4;          disp_step = 10
phi = 8;        momentum = 0.9
out_sz = 4

#inp_sz = 5;     lr = 0.0001 ## super small learning rate bc multiplicative effects
#N = 10;         epochs = 50
#theta = 8;      batchsize = 16
#B = 4;          disp_step = 10
#phi = 16;       momentum = 0.9
#out_sz = 2

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

def softmax_stable(v):
    return np.exp(v) / (1.0+np.exp(v))

def simple_task():
    if not (inp_sz==8 and out_sz==4): 
        print "Bad dims"
        sys.exit()
    a = float(np.random.randint(2))
    b = float(np.random.randint(2))
    c = float(np.random.randint(2))
    d = float(np.random.randint(2))
    e = float(np.random.randint(2))
    f = float(np.random.randint(2))
    g = float(np.random.randint(2))
    h = float(np.random.randint(2))
    return np.array([a,b,c,d,e,f,g,h], ndmin=2), np.array([\
            1.0 if (a==1 and b==1) else 0.0,\
            1.0 if (h==c and not d==e and f==0) else 0.0,\
            1.0 if (not a==c and not g==e and f==0) else 0.0,\
            1.0 if (a==f and not (a==1 or b==0)) else 0.0\
        ], ndmin=2)


def medium_task(n=100, I=inp_sz, O=out_sz):
    lx=[]; ly=[]
    for i in range(n):
        x_tmp = tuple([np.random.randint(-5,20) for _ in range(I)])
        lx.append(np.array(x_tmp, dtype='float32'))
        y_tmp = [softmax_stable(x_tmp[0] + x_tmp[1]), \
                 softmax_stable(2*x_tmp[1]-np.sum(x_tmp[2:])) ]
        while len(y_tmp)<O:
            y_tmp = y_tmp+[0]
        ly.append(np.array(y_tmp, dtype='float32'))
    return np.array(lx), np.array(ly)

def gate_variable(shape, myname, train):
    initial = tf.zeros(shape, dtype='float32')
    return tf.Variable(initial, name=myname, trainable=train)

def weight_variable(shape, myname, train):
    initial = tf.random_uniform(shape, minval = -0.3, maxval = 0.3, dtype='float32')
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
  #'M'      : weight_variable( (B, theta, phi), "Module", True), # trainable
  'M'      : weight_variable( (B, theta, phi), "Module", False), # NOT trainable
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
    accs = []
    for epoch in range(epochs):
        avg_cost = 0.0
        avg_grad = None
        for i in range(batchsize):
            R = np.random.randint(200)
            #x, y = train_I[R,:], np.expand_dims(train_O[R,:], 1)
            #x,y = dataset.train.next_batch(1)
            x,y = simple_task()
            x = np.expand_dims(x[0,:],0)

            ValsOfInterest = [optimizer, cost, grad_var]#+Weights_0.values()
            #for v in ValsOfInterest: print '\t',v
            _,c,g = sess.run(ValsOfInterest, feed_dict={x_var: x, y_var: y})
            avg_cost += c
            try: 
                avg_grad = avg_grad + g
            except:
                avg_grad = g
#            c = sess.run([optimizer, cost], feed_dict={x_var: x, y_var: y})
        print "Epoch", epoch, #"\tCost", avg_cost/batchsize\
                #, "\tAvg squared gradient", np.mean(avg_grad)**2,


        val_corr_pred = tf.equal(tf.argmax(this_model, 1), tf.argmax(y,1))
        #val_corr_pred = tf.equal(this_model, y)
        val_accuracy = tf.reduce_mean(tf.cast(val_corr_pred, "float"))
        val_accuracy = 1.0-(this_model-y)**2
        xv,yv = simple_task()
        val_cost = val_accuracy.eval({x_var: xv, y_var: yv})
#        print "\tVal acc", val_cost
        print ' --\t', [int(np.round(i)) for i in sess.run(   [this_model], \
                {x_var:xv, y_var:yv})[0][0]], [int(i) for i in y[0]]
        print '\t\t', [int(i) for i in xv.tolist()[0]]
#        accs.append(val_cost)
        accs.append(this_model.eval({x_var: xv, y_var: yv}))

    print "Opt finished."

    A = np.array(accs)
    print A.shape
    for a in range(A.shape[1]):
        plt.plot(A[:,a])
    plt.ylabel("Error")
    plt.xlabel("Epoch")
    plt.savefig(time.ctime())

#    corr_pred = tf.equal(tf.argmax(this_model, 1), tf.argmax(y,1))
#    accuracy = tf.reduce_mean(tf.cast(corr_pred, "float"))
#    x,y = np.squeeze(dataset.test.next_batch(1))
#    x_,y_=np.mean([simple_task() for _ in range(30)])
#    print "Accuracy simple:", np.mean( [accuracy.eval({x_var: x_, y_var: y_})
#                for r in range(30)])
#    print "Accuracy:", np.mean( [accuracy.eval({x_var: x, y_var: y})
#                for r in range(30)])

print "Done."
