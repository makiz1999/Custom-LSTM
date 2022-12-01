

import tensorflow as tf

def sigmoid_derivative(values):
        return values * (1 - values)


def tanh_derivative(values):
        return 1. - values ** 2
    
    
class LstmNode:
    
    def __init__(self, lstm_param, lstm_state):
       
        # print("\n---INIT NODE---")
        # store reference to parameters and to activations
        self.state = lstm_state
        self.param = lstm_param
        
        # non-recurrent input concatenated with recurrent input
        self.xc = None

    def bottom_data_is(self, x, s_prev=None, h_prev=None):
        
      #  print("\n---BOTTOM DATA IS---")

        # if this is the first lstm node in the network, initialize them
        if s_prev is None: s_prev = tf.zeros_like(self.state.s)
        if h_prev is None: h_prev = tf.zeros_like(self.state.h)
        
        # save data for use in backprop
        self.s_prev = s_prev
        self.h_prev = h_prev
        
        # x --> (n_samples X n_features)
        x = tf.transpose(x)
        x = tf.dtypes.cast(x, tf.float32)
  
        
        # xc --> ([n_features + n_neurons] X n_sample) --> compacting data
        xc = tf.concat([x, self.h_prev], axis=0)         
        xc = tf.reshape(xc, [len(xc), -1])
        xc = tf.Variable(xc)
   
        # h_prev --> (n_neurons X n_samples)
        self.h_prev = tf.reshape(self.h_prev, [len(h_prev), -1])
        self.s_prev = tf.reshape(self.s_prev, [len(s_prev), -1])
       
        # Compute LSTM four gate values
        self.state.g = tf.math.tanh(tf.matmul(self.param.wg, xc) + self.param.bg)
        self.state.i = tf.math.sigmoid(tf.matmul(self.param.wi, xc) + self.param.bi)
        self.state.f = tf.math.sigmoid(tf.matmul(self.param.wf, xc) + self.param.bf)
        self.state.o = tf.math.sigmoid(tf.matmul(self.param.wo, xc) + self.param.bo)
        
        # Computer LSTM two output values
        self.state.s = self.state.g * self.state.i + self.s_prev * self.state.f
        self.state.h = tf.math.tanh(self.state.s) * self.state.o
        
        # Dense layer by extending h --> h * wy + by  = y
        self.state.y = tf.matmul(self.param.wy, self.state.h)+ self.param.by
   
     
       # print (self.param.wy.shape)
       # print (self.state.h.shape) 
    
        self.xc = xc
        
        return self.state.y
    



    def top_diff_is(self, top_diff_h, top_diff_s, y_list, time_step):
        
       # print("\n---TOP DIFF IS---")
        # notice that top_diff_s is carried along the constant error carousel

        ds = self.state.o * top_diff_h + top_diff_s
        do = self.state.s * top_diff_h
        di = self.state.g * ds
        dg = self.state.i * ds
        df = self.s_prev * ds
        
       # print (self.state.y.shape)
       # print (y_list.shape)

        error = tf.transpose (self.state.y) - y_list # (n_samples,)

       # error = tf.reshape(error, [len(error), -1]) # into a vector (n_samples, 1)
        
        # diffs w.r.t. vector inside sigma / tanh function
        di_input = sigmoid_derivative(self.state.i) * di
        df_input = sigmoid_derivative(self.state.f) * df
        do_input = sigmoid_derivative(self.state.o) * do
        dg_input = tanh_derivative(self.state.g) * dg

        self.param.wi_diff = self.param.wi_diff + tf.matmul(di_input, tf.transpose(self.xc))
        self.param.wf_diff = self.param.wf_diff + tf.matmul(df_input, tf.transpose(self.xc))
        self.param.wo_diff = self.param.wo_diff + tf.matmul(do_input, tf.transpose(self.xc))
        self.param.wg_diff = self.param.wg_diff + tf.matmul(dg_input, tf.transpose(self.xc))
      
        # h --> ( n_neurons, n_samples)
        # error --> (n_samples X 1)
        
        # error_h is a scalar value 
        
        #print (error.shape)
       # print (self.state.h.shape)
        error_h = tf.matmul(tf.transpose (error), tf.transpose (self.state.h) )
        
        #print (error_h.shape,'error_h') # 1X16
       # print (self.param.wy_diff.shape, 'w_ydiff') # 1x16
        
       # Math expressions 
       # h_error (t) = (yhat(t) - y(t))* (h^T (t))
       # w_diff (t) = w_diff (t) + h_error (t)
        
        # Dense layer weight update
        self.param.wy_diff = self.param.wy_diff + error_h
        
        self.param.bi_diff = self.param.bi_diff + di_input
        self.param.bf_diff = self.param.bf_diff + df_input
        self.param.bo_diff = self.param.bo_diff + do_input
        self.param.bg_diff = self.param.bg_diff + dg_input
        
        #Dense layer bias term update
        print(error.shape)
        self.param.by_diff = self.param.by_diff + error
        
        
   

        # compute bottom diff
        dxc = tf.zeros_like(self.xc)
        dxc += tf.matmul(tf.transpose(self.param.wi), di_input)
        dxc += tf.matmul(tf.transpose(self.param.wf), df_input)
        dxc += tf.matmul(tf.transpose(self.param.wo), do_input)
        dxc += tf.matmul(tf.transpose(self.param.wg), dg_input)

        # save bottom diffs
        self.state.bottom_diff_s = ds * self.state.f
        self.state.bottom_diff_h = dxc[self.param.n_features:]
