"""
3/14/17   Morgan Bryant

Arguments/Parameters: Invoke as 'python ANN.py -h' or 'ipython ANN.py -- -h'

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time, random, sys, os, argparse, time
from datetime import datetime
startTime = datetime.now()

# This is a simple ANN implemented to compare to ModIAN results.

inp_sz = 8 ;    lr = .0001## super small learning rate bc multiplicative effects
l1 = 6;        epochs = 10000
l2 = 6;        batchsize = 1
out_sz = 4 ;    disp_step = 10


print "Parameters inp_sz, l1, l2, out_sz, lr, epochs, batchsize:"
print inp_sz,l1, l2 ,out_sz,lr,epochs,batchsize

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
def simple_task0():
    a = float(np.random.randint(2))
    b = float(np.random.randint(2))
    return [np.array([a,b],ndmin=2), np.array([[1.0,0.0] if 
        (not a==b) else [0.0,1.0] ],ndmin=2)]


def simple_task1():
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
            1.0 if (not a==c and not g==e and f==1) else 0.0,\
            1.0 if (a==f and not (a==1 or b==0)) else 0.0\
        ], ndmin=2)

def simple_task2():
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
#    MIN = -3.2/nvars
#    MAX = 9.6/nvars
    MIN = -10.0/nvars
    MAX = 10.0/nvars
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

def model(inp, w):
  with tf.variable_scope("network", reuse=None):
    inp_r = tf.reshape(inp, (inp_sz, 1))
    print inp_r.get_shape(),w['W_I_L1'].get_shape(),  w['b_I_L1'].get_shape()

    Z1  = tf.add(tf.matmul(w['W_I_L1'], inp_r), w['b_I_L1'])
    Z_1 = tf.nn.relu(Z1)
    Z2  = tf.add(tf.matmul(w['W_L1_L2'], Z_1), w['b_L1_L2'])
    Z_2 = tf.nn.relu(Z2)
    Zout= tf.add(tf.matmul(w['W_L2_O'], Z_2), w['b_L2_O'])

    return tf.reshape(Zout, (1,out_sz))
    #return tf.nn.softmax(tf.reshape(Zout, (1,out_sz)))

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
    'W_I_L1' : weight_variable((l1,inp_sz), "Weights_Inp_to_L1",True),
    'b_I_L1' : weight_variable((l1,1),      "biases_Inp_to_L1", False),
    'W_L1_L2': weight_variable((l2,l1),     "Weights_L1_to_L2", True),
    'b_L1_L2': weight_variable((l2,1),      "biases_L1_to_L2", False),
    'W_L2_O' : weight_variable((out_sz,l2), "Weights_L2_to_O", True),
    'b_L2_O' : weight_variable((out_sz,1), "biases_L2_to_O", False),
  }
  P={}
  for w in Weights_0:
      W=Weights_0[w]
      meanW = tf.reduce_mean(W)
      varW = tf.reduce_mean((W-meanW)**2)
      P[w] = tf.Print(W,[meanW, varW], summarize=min(np.prod(W.get_shape()),10))
  
  x_var = tf.placeholder("float32", [1,inp_sz])
  y_var = tf.placeholder("float32", [1,out_sz])
  this_model = model(x_var, Weights_0)
  
#  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits\
#          (logits = this_model, labels = y_var))
#  cost = tf.nn.softmax_cross_entropy_with_logits(logits = this_model, labels = y_var)
#  cost = tf.nn.l2_loss(this_model - y_var)
  cost = tf.reduce_mean(tf.pow(y_var-this_model, 2))
  optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr)
  grads = optimizer.compute_gradients(y_var)
  grad_var = tf.gradients(cost, Weights_0.values())[0]
  optimizer = optimizer.minimize(cost)
  ValsOfInterest = [optimizer, cost, grad_var]#+Weights_0.values()
  
  print "SHAPE:",this_model.get_shape(), y_var.get_shape()
  init = tf.initialize_all_variables()
  G.finalize()
  
  TRAINSET = [simple_task1() for _ in range(600)]
  VALSET = [simple_task1() for _ in range(100)]
  print "Task: simple_task1."
  accs = []
  ACCS = []
  
  with tf.Session() as sess:
    sess.run(init)
    curtime = time.time()
    avg_time = 0.0
    print "y_var.get_shape()", (y_var).get_shape()
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
            accs += [c]
            try: 
                avg_grad = avg_grad + g
            except:
                avg_grad = g
#            c = sess.run([optimizer, cost], feed_dict={x_var: x, y_var: y})
        if not epoch%100: print "Epoch", epoch, #"\tCost", avg_cost/batchsize\
                #, "\tAvg squared gradient", np.mean(avg_grad)**2,
        t = time.time()
        if not epoch%100: print '\ttime:', '%.3f'%( t-curtime ),
        avg_time += (t-curtime)
        if not epoch%100: print ' avg:',  '%.3f'%( avg_time/(1+epoch) ),'s',
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
        #CURACC = np.array([yv[0,:],this_model.eval({x_var: xv, y_var: yv})[0,:]]).T
        CURACC = np.array([yv[0,:],\
            this_model.eval({x_var: xv})[0,:]]).T
        ACCS.append( np.mean( np.power((CURACC[:,0] - CURACC[:,1]), 2)) )
        #print yv, accs[-1], (yv-accs[-1])
#        if not epoch%10: print '\t',xv,yv, 
        #if not epoch%1: print '\n\tErr:', ['%.5f'%a for a in accs[-1][0]]
        #if not epoch%1: print '\tErr:', ['%.4f'%a for a in accs[-1][0]]
        if not epoch%100: print '\tErr:', CURACC, accs[-1], ACCS[-1]
        #tf.reset_default_graph()
    for w in Weights_0:
        print '\nWeights',w,':'
        sess.run(P[w])
    print '\n\n'
 
print "Opt finished."
#
A = np.array(accs)
print A.shape
print_epoch=20
plt.subplot(111).set_yscale("log")
try:
 for a in range(A.shape[2]):
  s = 'Logic task #'+str(a)
  print s
  A_ = [np.mean(A[print_epoch*index:(print_epoch*(1+index)-1),0,a], axis=0) 
          for index in range(A.shape[0]/print_epoch)]
  plt.plot(A_, label=s)
  #print (A[:,a*print_epoch:((a+1)*print_ep-1)]).shape, A_.shape
except:
    plt.plot(accs[0::10])
    plt.plot(ACCS[0::10])
plt.title("Trial begun at "+str(startTime))
plt.ylabel("Error")
plt.xlabel("Epoch x "+str(print_epoch))
plt.legend()
plt.savefig(time.ctime())
plt.close()
#
#    corr_pred = tf.equal(tf.argmax(this_model, 1), tf.argmax(y,1))
#    accuracy = tf.reduce_mean(tf.cast(corr_pred, "float"))
#    x,y = np.squeeze(dataset.test.next_batch(1))
#    x_,y_=np.mean([simple_task() for _ in range(30)])
#    print "Accuracy simple:", np.mean( [accuracy.eval({x_var: x_, y_var: y_})
#                for r in range(30)])
#    print "Accuracy:", np.mean( [accuracy.eval({x_var: x, y_var: y})
#                for r in range(30)])

print "Done."
