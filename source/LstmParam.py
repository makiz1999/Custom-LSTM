
import tensorflow as tf

class LstmParam:
    
    def __init__(self, n_input, n_neurons, n_features, n_output=1):
        
      #  print("\n---INIT PARAM---")
        
        self.n_input = n_input
        self.n_neurons = n_neurons
        self.n_features = n_features
        concat_len = n_features + n_neurons
        self.concat_len = concat_len
        
        # weight matrices
        self.wg = tf.Variable(tf.random.truncated_normal([n_neurons, concat_len], stddev=0.01), name='wg')
        self.wi = tf.Variable(tf.random.truncated_normal([n_neurons, concat_len], stddev=0.01), name='wi')
        self.wf = tf.Variable(tf.random.truncated_normal([n_neurons, concat_len], stddev=0.01), name='wf')
        self.wo = tf.Variable(tf.random.truncated_normal([n_neurons, concat_len], stddev=0.01), name='wo')
        self.wy = tf.Variable(tf.random.truncated_normal([n_output, n_neurons], stddev=0.01), name='wy')
        
        # bias terms
        self.bg = tf.Variable(tf.constant(0.0, shape=([n_neurons, 1])), name='bg')
        self.bi = tf.Variable(tf.constant(0.0, shape=([n_neurons, 1])), name='bi')
        self.bf = tf.Variable(tf.constant(1.0, shape=([n_neurons, 1])), name='bf')
        self.bo = tf.Variable(tf.constant(0.0, shape=([n_neurons, 1])), name='bo')
        self.by = tf.Variable(tf.constant(0.0, shape=([n_output, 1])), name='by')
        
        # diffs (derivative of loss function w.r.t. all parameters)
        # self.wg_diff = tf.zeros((n_neurons, concat_len))
        self.wg_diff = tf.Variable(tf.constant(0.0, shape=([n_neurons, concat_len])))
        self.wi_diff = tf.Variable(tf.constant(0.0, shape=([n_neurons, concat_len])))
        self.wf_diff = tf.Variable(tf.constant(0.0, shape=([n_neurons, concat_len])))
        self.wo_diff = tf.Variable(tf.constant(0.0, shape=([n_neurons, concat_len])))
        self.wy_diff = tf.Variable(tf.constant(0.0, shape=([n_output, n_neurons])))

        # self.bg_diff = tf.zeros(n_neurons)
        self.bg_diff = tf.Variable(tf.constant(0.0, shape=([n_neurons, 1])), name='bg_diff')
        self.bi_diff = tf.Variable(tf.constant(0.0, shape=([n_neurons, 1])), name='bi_diff')
        self.bf_diff = tf.Variable(tf.constant(0.0, shape=([n_neurons, 1])), name='bf_diff')
        self.bo_diff = tf.Variable(tf.constant(0.0, shape=([n_neurons, 1])), name='bo_diff')
        self.by_diff = tf.Variable(tf.constant(0.0, shape=([n_output, 1])), name='bo_diff')


    def apply_diff(self, lr=0.05):
        
        print("\n---APPLY DIF---")
        self.wg = self.wg - (lr * self.wg_diff)
        self.wi = self.wi - (lr * self.wi_diff)
        self.wf = self.wf - (lr * self.wf_diff)
        self.wo = self.wo - (lr * self.wo_diff)
        
        # --- UPDATE Wy ---
        self.wy = self.wy - (lr*self.wy_diff)
        

        self.bg = self.bg - (lr * self.bg_diff)
        # print("apply diff bg")
        # print(self.bg.shape)
        self.bi = self.bi - (lr * self.bi_diff)
        self.bf = self.bf - (lr * self.bf_diff)
        self.bo = self.bo - (lr * self.bo_diff)
        # reset diffs to zero
        self.wg_diff = tf.zeros_like(self.wg)
        self.wi_diff = tf.zeros_like(self.wi)
        self.wf_diff = tf.zeros_like(self.wf)
        self.wo_diff = tf.zeros_like(self.wo)
        self.wy_diff = tf.zeros_like(self.wy)
      
        # biases
        self.bg_diff = tf.zeros_like(self.bg)
        self.bi_diff = tf.zeros_like(self.bi)
        self.bf_diff = tf.zeros_like(self.bf)
        self.bo_diff = tf.zeros_like(self.bo)

        self.by_diff = tf.zeros_like(self.by)