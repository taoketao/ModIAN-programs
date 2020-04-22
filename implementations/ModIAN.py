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
#import tf_mnist_loader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time, random, sys, os, argparse, time
from datetime import datetime
startTime = datetime.now()

#dataset = tf_mnist_loader.read_data_sets("mnist_data", one_hot=True)

inp_sz = 8;     lr = 0.0003 ## super small learning rate bc multiplicative effects
N = 8;          epochs = 3000
theta = 8;      batchsize = 16
B = 4;          disp_step = 10
phi = 8;        momentum = 0.9
out_sz = 14

print "Parameters inp_sz,N,theta,B,phi,out_sz,lr,epochs,batchsize:"
print inp_sz,N,theta,B,phi,out_sz,lr,epochs,batchsize

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

def simple_task2():
    if not (inp_sz==8 and out_sz==14): 
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
            1.0 if (a and b) else 0.0,\
            1.0 if (g or d) else 0.0,\
            1.0 if (c or e or f or h) else 0.0,\
            1.0 if (c==d and e==f) else 0.0,\
            1.0 if (c==d or e==f) else 0.0,\
            1.0 if not (g==h) else 0.0,\
            1.0 if g==(h or a) else 0.0,\
            1.0 if (a==b and a==c) else 0.0,\
            1.0 if (d or not e) else 0.0,\
            1.0 if (a or b) == (c and d) else 0.0, \
            1.0 if (not (a or b)) == (c and d) else 0.0, \
            1.0 if (a and b==a) or (c and b==c) else 0.0, \
            1.0 if g==h else 0.0,\
            1.0 if not f==h else 0.0, \
        ], ndmin=2)


# see https://en.wikipedia.org/wiki/Linearity:
# linear boolean functions



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

def gate_variable(shape, myname, train):  # want 
    initial = tf.zeros(shape, dtype='float32')
    return tf.Variable(initial, name=myname, trainable=train)

def weight_variable(shape, myname, train):
    nvars = 1
    for s in shape:
        nvars = nvars * s
    MIN = -3.2/nvars
    MAX = 9.6/nvars
    initial = tf.random_uniform(shape, minval = MIN, maxval = MAX, dtype='float32')
    print "weight var",myname,"initialized uni:", '%.3f'%(MIN), '%.3f'%(MAX)
    return tf.Variable(initial, name=myname, trainable=train)

def xavier_variable(shape, myname, train):
    nvars = 1
    for s in shape:
        nvars = nvars * s
    xav = (12.0/nvars)**0.5
    MIN = 0
    MAX = xav
    initial = tf.random_uniform(shape, minval = MIN, maxval = MAX, dtype='float32')
    print "weight var",myname,"initialized by quasi-2x-xavier: uni(",\
            '%.3f'%(MIN),'%.3f'%(MAX),'), xavier=','%.3f'%(xav)
    return tf.Variable(initial, name=myname, trainable=train)

def model(inp, w, trainable_module=False, identity_x_th=False):
  with tf.variable_scope("network", reuse=None):

    inp_r = tf.reshape(inp, (inp_sz, 1))
    W_I_XB_r = tf.reshape(w['W_I_XB'], (N*B, inp_sz))
    b_I_XB_r = tf.reshape(w['b_I_XB'], (N*B,1))
    W_I_Xth_r = tf.reshape(w['W_I_Xth'], (N*theta, inp_sz))
    b_I_Xth_r = tf.reshape(w['b_I_Xth'], (N*theta,1))
    if trainable_module:
        M_th_B = tf.reshape(w['M'], (B * phi, theta))
        Mb1 = tf.reshape(w['Mb1'], (B*phi,1,))
        Mb2 = w['Mb2']
    else:
        M_th_B = tf.reshape(w['M_t'], (B * phi, theta))
        Mb1 = tf.reshape(w['Mb1_t'], (B*phi,1,))
        Mb2 = w['Mb2_t']
    print "Is module M trainable?", 'YES' if trainable_module else 'NO'
    W_Y_X_new = tf.reshape(w['W_Y_X1'], (out_sz, N*phi))
    b_Y_X_new = tf.reshape(w['b_Y_X1'], (out_sz, 1))

    X_B  = tf.nn.softmax(tf.matmul(W_I_XB_r, inp_r) + b_I_XB_r)
    if identity_x_th and inp_sz==N*theta:
        x_th = tf.reshape(inp_r, (N, theta, 1))
    elif not identity_x_th:
        X_th = tf.matmul(W_I_Xth_r, inp_r) + b_I_Xth_r
        x_th = tf.reshape(X_th, (N, theta, 1))
    else:
        print "Bad dims:", inp_sz,'=/=',N*theta
        sys.exit()
    x_b = tf.reshape(X_B, (N, B, 1))

    def module_operation_TH(x_th):
        return tf.matmul(M_th_B, x_th) + Mb1
    def module_operation_B(i):
        Z_slice = tf.reshape(tf.slice(Zr, [i,0,0],[1,-1,-1]), (phi,B))
        xb_slice = tf.reshape(tf.slice(x_b, [i,0,0],[1,-1,-1]), (B,1))
        return tf.matmul(Z_slice, xb_slice) + Mb2

    Z = tf.map_fn(module_operation_TH, x_th, dtype='float32')
    Zr = tf.reshape(Z, (N,  phi, B,))
    Y = tf.map_fn(module_operation_B, tf.range(N), dtype='float32') 
    Yr = tf.reshape(Y, (N*phi,1))
    X_new = tf.nn.relu( tf.matmul(W_Y_X_new, Yr) + b_Y_X_new)
    return tf.reshape(X_new, (1,out_sz))

# I -> X_t.  I -> X_B.  sigma( (MX_t)sigma(X_B) ) -> Y.  Y -> X'.
#   inp_sz = 5;     lr = 0.001
#   N = 10;         epochs = 10
#   theta = 8;      batchsize = 16
#   B = 4;          disp_step = 10
#   phi = 16        momentum = 0.9
#   out_sz = 2

G = tf.Graph()
with G.as_default():
  Weights_0 = {
    'W_I_Xth': weight_variable((inp_sz, N, theta), "Weights_Inp_to_Xtheta",True),
    'b_I_Xth': weight_variable((1, N, theta), "biases_Inp_to_Xtheta", True),
    'W_I_XB' : gate_variable((inp_sz, N, B), "Weights_Inp_to_XB", True),
    'b_I_XB' : gate_variable((1, N, B), "biases_Inp_to_XB", True),
    'M_t'    : weight_variable( (B, theta, phi), "Module_t", True), # trainable
    'Mb1_t'  : weight_variable( (B, 1, phi), "M_bias_theta_t", True), # trainable
    'Mb2_t'  : weight_variable( (phi, 1), "M_bias_phi_t", True), # trainable
    'M'      : weight_variable( (B, theta, phi), "Module", False), # NOT trainable
    'Mb1'    : weight_variable( (B, 1, phi), "M_bias_theta", False), # NOT trainable
    'Mb2'    : weight_variable( (phi, 1), "M_bias_phi", False), # NOT trainable
    'W_Y_X1' : weight_variable( (N, phi, out_sz), "Weights_Y_to_Xprime", True),
    'b_Y_X1' : weight_variable( (1, 1, out_sz), "biases_Y_to_Xprime", True),
    #  biases added to inp X -> theta, X -> B, M -> Y, and Y -> X'.
  }
  
  x_var = tf.placeholder("float32", [1,inp_sz])
  y_var = tf.placeholder("float32", [1,out_sz])
  this_model = model(x_var, Weights_0, trainable_module=False, identity_x_th=False)
  
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits\
          (logits = this_model, labels = tf.to_float(y_var)))
  optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr)
  grads = optimizer.compute_gradients(y_var)
  grad_var = tf.gradients(cost, Weights_0.values())[0]
  optimizer = optimizer.minimize(cost)
  ValsOfInterest = [optimizer, cost, grad_var]#+Weights_0.values()
  
  print "SHAPE:",this_model.get_shape(), y_var.get_shape()
  init = tf.initialize_all_variables()
  G.finalize()
  
  TRAINSET = [simple_task2() for _ in range(600)]
  VALSET = [simple_task2() for _ in range(100)]
  print "Task: simple_task2."
  accs = []
  
  with tf.Session() as sess:
    sess.run(init)
    curtime = time.time()
    avg_time = 0.0
    for epoch in range(epochs):
        avg_cost = 0.0
        avg_grad = None
        for i in range(batchsize):
            R = np.random.randint(600)
            #x, y = train_I[R,:], np.expand_dims(train_O[R,:], 1)
            #x,y = dataset.train.next_batch(1)
            #x,y = simple_task()
            x,y=TRAINSET[R]
            x = np.expand_dims(x[0,:],0)
            #for v in ValsOfInterest: print '\t',v
            _,c,g = sess.run(ValsOfInterest, feed_dict={x_var: x, y_var: y})
            avg_cost += c
            try: 
                avg_grad = avg_grad + g
            except:
                avg_grad = g
#            c = sess.run([optimizer, cost], feed_dict={x_var: x, y_var: y})
        if not epoch%10: print "Epoch", epoch, #"\tCost", avg_cost/batchsize\
                #, "\tAvg squared gradient", np.mean(avg_grad)**2,
        t = time.time()
        if not epoch%10: print '\ttime:', '%.3f'%( t-curtime ),
        avg_time += (t-curtime)
        if not epoch%10: print ' avg:',  '%.3f'%( avg_time/(1+epoch) ),'s',
        curtime=t

        #val_corr_pred = tf.equal(tf.argmax(this_model, 1), tf.argmax(y,1))
        #val_corr_pred = tf.equal(this_model, y)
        #val_accuracy = tf.reduce_mean(tf.cast(val_corr_pred, "float"))
        #xv,yv = simple_task()
        xv,yv = VALSET[np.random.randint(100)]
        #val_cost = val_accuracy.eval({x_var: xv, y_var: yv})
#            print "\tVal acc", val_cost
#        X = [int(np.round(i)) for i in sess.run(   [this_model], \
#                {x_var:xv, y_var:yv})[0][0]]
#        print 'xxx',X, [int(i) for i in y[0]]
#            print ' --\t', [int(np.round(i)) for i in sess.run(   [this_model], \
#                    {x_var:xv, y_var:yv})[0][0]], [int(i) for i in y[0]]
#        print '\t\t', [int(i) for i in xv.tolist()[0]]
        #accs.append(X)
        #accs.append(val_cost)
        accs.append(this_model.eval({x_var: xv, y_var: yv}))
        #if not epoch%10: print '\n\tErr:', ['%.5f'%a for a in accs[-1][0]]
        if not epoch%10: print '\tErr:', ['%.4f'%a for a in accs[-1][0]]
        #tf.reset_default_graph()

print "Opt finished."

A = np.array(accs)
print A.shape
print_epoch=20
plt.subplot(111).set_yscale("log")
for a in range(A.shape[2]):
  s = 'Logic task #'+str(a)
  print s
  A_ = [np.mean(A[print_epoch*index:(print_epoch*(1+index)-1),0,a], axis=0) 
          for index in range(A.shape[0]/print_epoch)]
  plt.plot(A_, label=s)
  #print (A[:,a*print_epoch:((a+1)*print_ep-1)]).shape, A_.shape
plt.title("Trial begun at "+str(startTime))
plt.ylabel("Error")
plt.xlabel("Epoch x "+str(print_epoch))
plt.legend()
plt.savefig(time.ctime())
plt.close()

#    corr_pred = tf.equal(tf.argmax(this_model, 1), tf.argmax(y,1))
#    accuracy = tf.reduce_mean(tf.cast(corr_pred, "float"))
#    x,y = np.squeeze(dataset.test.next_batch(1))
#    x_,y_=np.mean([simple_task() for _ in range(30)])
#    print "Accuracy simple:", np.mean( [accuracy.eval({x_var: x_, y_var: y_})
#                for r in range(30)])
#    print "Accuracy:", np.mean( [accuracy.eval({x_var: x, y_var: y})
#                for r in range(30)])

print "Done."
