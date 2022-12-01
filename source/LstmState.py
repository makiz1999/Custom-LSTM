
import tensorflow as tf

class LstmState:
    
    def __init__(self, n_input, n_neurons, n_features, n_output=1):
        
       # print("\n---INIT STATE---")
    
        # Initializing gates
        self.g = tf.Variable(tf.constant(0.0, shape=([n_neurons, n_input])))
        self.i = tf.Variable(tf.constant(0.0, shape=([n_neurons, n_input])))
        self.f = tf.Variable(tf.constant(0.0, shape=([n_neurons, n_input])))
        self.o = tf.Variable(tf.constant(0.0, shape=([n_neurons, n_input])))
        
        # Initializing two top outputs
        self.s = tf.Variable(tf.constant(0.0, shape=([n_neurons, n_input])))
        self.h = tf.Variable(tf.constant(0.0, shape=([n_neurons, n_input])))
        
        # Initializing the final output 
        self.y = tf.Variable(tf.constant(0.0, shape=([n_output, 1])))
        
        # Initializing two bottom outputs difference states
        self.bottom_diff_h = tf.zeros_like(self.h)
        self.bottom_diff_s = tf.zeros_like(self.s)



