import numpy as np
import tensorflow as tf

from source import LstmState 
from source import LstmNode

class LossLayer:
    """
    Computes square loss with first element of hidden layer array.
    """
    @classmethod
    def loss(self, pred, label):
            return (pred - label) ** 2
        

    @classmethod
    def bottom_diff(self, pred, label):
            
            diff = np.zeros_like(pred)
            
            for n_sam in range(len(pred[0])):
                
                #diff[0][n_sam] = 2 * (pred[0][n_sam] - label[n_sam])
                
                        diff[:, n_sam] = 2 * (pred[:, n_sam] - label[n_sam])* self.lstm_node_list[tm_idx].param.wy
            
            
           # diff[0] = 2 * (pred[0] - label)
            
            return diff
        
        
class LstmNetwork():

    def __init__(self, lstm_param):
        
        #print("\n---INIT NETWORK---")
       
        self.lstm_param = lstm_param
        self.lstm_node_list = []
        
        # list of input time sequence
        self.x_list = []

    def x_list_clear(self):
        
        print("\n---X LIST CLEAR---")
        
        self.x_list = []

    
    def x_list_time_loop(self, x):
       
        # x --> (n_samples X n_features)
        # x_list --> maximum n_steps dimensional 
        
        self.x_list.append(x)
        
        
        # in case an additional time point is available consider that ...
        if len(self.x_list) > len(self.lstm_node_list):

            
            # create a state object for each time step using input parameters (weights and biases)
            # state variables are g, f, o, i, s, h, and y
            lstm_state = LstmState.LstmState(self.lstm_param.n_input, self.lstm_param.n_neurons,
                                   self.lstm_param.n_features)
            
            
            # create an lstm node object combining lstm_param and lstm_state
            self.lstm_node_list.append(LstmNode.LstmNode(self.lstm_param, lstm_state))

            
        # get index of the current time point
        tm_idx = len(self.x_list) - 1
        
        if tm_idx == 0:  # Estimate y_pred for the first time sample (x at t=0)
       
            y_pred = self.lstm_node_list[tm_idx].bottom_data_is(x)
    
        
        else: # for subsequent time points after t = 0
            
            s_prev = self.lstm_node_list[tm_idx - 1].state.s # long-term memory

            h_prev = self.lstm_node_list[tm_idx - 1].state.h # short-term memory

            y_pred = self.lstm_node_list[tm_idx].bottom_data_is(x, s_prev, h_prev)
            
            

        return y_pred
    
    
    
    
    def y_list_is(self, y_list, loss_layer):
        
        y_list = np.array(y_list)
       
       # print("\n---Y LIST IS---")
        """
        Updates diffs by setting target sequence
        with corresponding loss layer.
        Will *NOT* update parameters.  
        To update parameters, call self.lstm_param.apply_diff()
        """
        print(len(y_list[0]))
        assert len(y_list[0]) == len(self.x_list)

        tm_idx = len(self.x_list) - 1

        loss = 0
        loss_global = 0
        
        # Predicted y at tm_idx --> (n_output X n_samples)
        
        y_hat = self.lstm_node_list[tm_idx].state.y
        print(y_hat.shape)
        
        # Y actual at time tm_idx -- ground truth
        y_act = y_list[:, tm_idx,:]
         
        # diff_h has the same dimension as h, initializing with zeros    
        diff_h = np.zeros_like(self.lstm_node_list[tm_idx].state.h)
        
       # print ('yhat',y_hat.shape)
       # print ('yact',y_act.shape)
       # print ('ydiff',diff_h.shape)
   
        # iterating over individual samples 
        for sam_idx in range(len(y_list)):
            
            #d_h = 2* (yhat-yact)*Wy
        
            diff_h[:, sam_idx] = 2 * (y_hat[:, sam_idx] - y_act[sam_idx])* self.lstm_node_list[tm_idx].param.wy
            
            
            # y[0][sam_idx] --> 0 is to select only one output since n_output =1
            # For multi-dimensional output, n_output>1, we have to check this carefully
            loss += loss_layer.loss(self.lstm_node_list[tm_idx].state.y[0][sam_idx], y_list[sam_idx, tm_idx,:])
          
        
        loss_global = loss / len(y_list)
        
        
        diff_s = tf.zeros(self.lstm_param.n_neurons)  # (n_neurons,)
        diff_s = tf.reshape(diff_s, [len(diff_s), -1]) # (n_neurons,1)
        
        # diff values are getting processed inside the LSTM black-box
        self.lstm_node_list[tm_idx].top_diff_is(diff_h, diff_s, y_list[:,tm_idx,:], tm_idx)
        
        tm_idx -= 1

        ### ... following nodes also get diffs from next nodes, hence we add diffs to diff_h
        ### we also propagate error along constant error carousel using diff_s
        
        while tm_idx >= 0:
        
            
            for sam_idx in range(len(y_list)):
                
                loss += loss_layer.loss(self.lstm_node_list[tm_idx].state.y[0][sam_idx], y_list[sam_idx, tm_idx,:])
                
            
            loss_global += loss / len(y_list)
            
            # we are computing diff_h at time t = ti
          #  diff_h =  loss_layer.bottom_diff(self.lstm_node_list[tm_idx].state.h, y_list[:, tm_idx])
            
            
            # Adding diff_h from current and previous time steps
            diff_h = diff_h + self.lstm_node_list[tm_idx + 1].state.bottom_diff_h
            
            # Updating (not adding) s (long-term memory) 
            diff_s =  self.lstm_node_list[tm_idx + 1].state.bottom_diff_s
           
            
            # diff values are getting processed inside the LSTM black-box
            self.lstm_node_list[tm_idx].top_diff_is(diff_h, diff_s, y_list[:,tm_idx], tm_idx)
        
            tm_idx -= 1

        
        loss_global = loss_global / len(self.x_list) 

        return loss_global
    
    
    
    
