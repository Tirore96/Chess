import tensorflow as tf
import sys
import boardclass
import random
import time
import numpy as np
import pdb
import os
import pickle
import glob
from multiprocessing import Pool

input_height = 32
input_width = 26
input_size = input_height * input_width
num_squares = 8 *8 
filter_size1 = 2
num_filters1 = 3

filter_size2 = 2
num_filters2 = 6
last_connected_output = 2048

fc_size = 128

batch_size = 1
learning_rate = 1e-5
r_worse_than_q = 2
k = 15
r_priority = k/2
k_p = 20
weights_val = 0.5
bias_val = 0.5

class Model:
    def __init__(self):
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
    
    def run_session(self,file_fetcher,index=-1,fetch_index=False):
        iterations = file_fetcher.max_id
        start_time = time.time()
        if fetch_index:
            p_batch,q_batch,r_batch = file_fetcher.fetch_index(index)
        for i in range(len(p_batch)):

     #       else:
     #           p_batch,q_batch,r_batch = file_fetcher.fetch()           
            p_in = p_batch[i].reshape(1,832)
            q_in = q_batch[i].reshape(1,832)           
            r_in = r_batch[i].reshape(1,832)               
            feed_dict_train = {p:p_in,q:q_in,r:r_in}
            
            self.session.run(optimizer,feed_dict=feed_dict_train)
            
            if 0 == iterations%20 :
                score = self.session.run(p_val,feed_dict={p:p_batch})
                print("score P: {}".format(score[0]))
                
                score = self.session.run(q_val,feed_dict={q:q_batch})
                print("score Q: {}".format(score[0]))
                
                score = self.session.run(r_val,feed_dict={r:r_batch})
                print("score R: {}".format(score[0]))

        end_time = time.time()
        print("Time used to train model: " + str(end_time-start_time))
        
    def save(self,path):
        #self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        path_ret = saver.save(self.session,path)
        print("Model saved at {}".format(path_ret))


    def restore(self,path):
        self.session = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.session,path)
        
    def delete_model_files(self,folder_path,prefix):
        os.remove("{}/checkpoint".format(folder_path))
        extended_paths = "{0}/{1}*".format(folder_path,prefix)
                
        for filename in glob.glob(extended_paths):
            os.remove(filename)

            #.model_checkpoint_path
            #imported_meta = tf.train.import_meta_graph(self.path)#"model_final.meta")
            #imported_meta.restore(self.session,tf.train.latest_checkpoint("../"))
    #
    def evaluate(self,p_cur):
        p_cur = p_cur.reshape((1,832))
        score = self.session.run(p_val,feed_dict={p:p_cur})
        return score
    
    def close_session(self):
        self.session.close()

#credit to Hvass Laboratories
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=weights_val))

def new_biases(length):
    return tf.Variable(tf.constant(bias_val,shape=[length]))

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_falt = tf.reshape(layer,[-1,num_features])
    return layer_falt, num_features


def create_index_intervals(limit,split):
    if limit < split:
        return [(0,0)]
    retarr = []
    high_end = 0
    offset = 0
    pool_size = int(limit/split)
    max_offset = pool_size * split
    while offset < max_offset:
        retarr.append((offset,pool_size))
        offset = offset + pool_size
    return retarr

def gen_pqr_pool(path,limit,split):
    move_list_2d_curated = boardclass.file_to_move_lists(path,groundtruth=True)
    index_intervals = create_index_intervals(limit,split)
    arr_inputs = [(move_list_2d_curated,index_intervals[i][0],index_intervals[i][1],i) for i in range(len(index_intervals))]
    p = Pool()
    retvals = p.starmap(gen_pqr_tuples,arr_inputs)
 #   p = []
 #   q = []
 #   r = []
 #   for i in retvals:
 #       p.append(i[0])
 #       q.append(i[1])       
 #       r.append(i[2])           
 #   return p,q,r

def gen_pqr_tuples(move_list_2d_curated,offset,elements,id_num):
    p = []
    q = []
    r = []

    curated_len = len(move_list_2d_curated)
#    if limit == -1:
#        limit = curated_len
        
    counter =0
    for i in range(offset,offset+elements):
        all_states_encoded = []
        board = boardclass.ChessBoard(-1,only_board=True)
        encoded_state = board.one_hot_encode_board()
        all_states_encoded.append(encoded_state)
        for a_move in move_list_2d_curated[i]:
            r_board_encoded,board = encode_board_after_random_move(board)
            board.make_move(a_move)
            encoded_state = board.one_hot_encode_board()
            all_states_encoded.append(encoded_state)
            r.append(r_board_encoded)

            if not board.made_move: 
                print("ERROR did not make move in gen_pqr_tupes")
                return a_move_list
        p.extend(all_states_encoded[:-1])
        q.extend(all_states_encoded[1:])
        
     
        counter = counter + 1
#        if counter % 50 == 0:
#            print("was here")
#            print("{} out of {}".format(counter,limit))

    bin_file = open("pickled/pickled_train_data.bin{}".format(id_num),mode='wb+')
    pickle.dump([p,q,r],bin_file)
       
    print("{} is done now".format(id_num))
#    return p,q,r
            
            
def encode_board_after_random_move(b):
    moves = b.generate_legal_moves()
    moves_len = len(moves)
    index = random.randint(0,moves_len-1)
    b.make_move(moves[index])
    one_hot = b.one_hot_encode_board()
    b.unmake_move()
    return one_hot,b
        

def file_to_training_data(path):
    train_x = []
    train_y = []
    file = open(path,"r")
    file_string = file.read()
    replaceable_chars = ["\n"]
    for i in replaceable_chars:
        file_string = file_string.replace(i,"")
    file_array = file_string.split(" ")
    board = ChessBoard(-1)
    count = 0
    for i in file_array:
        if i == "None":
            board = ChessBoard(-1)
            
            continue
        if i == "":
            break
        
        train_x.append(board.one_hot_encode_board())
        board.make_move(i[:4])
        train_y.append(board.one_hot_encode_move(i[:4]))
        count = count + 1
    train_x = np.asarray(train_x,dtype=np.float)
    train_y = np.asarray(train_y,dtype=np.float)   
    return train_x,train_y

class TrainingData:
    def __init__(self,train_x,train_y):
        self.train_x = np.asarray(train_x)
        self.train_y = np.asarray(train_y)
        self.x_shape = self.train_x.shape
        self.y_shape = self.train_y.shape       
        self.x_dtype = self.train_x.dtype
        self.y_dtype = self.train_y.dtype
        self.index  = 0
        self.len = len(self.train_x)
        if len(self.train_x) != len(self.train_y):
            raise Exception("x and y are not same length")
    
    def next_batch(self,size):
        if self.index + size >= self.len:
            print("size is larger than the remaining batch")
            return np.zeros(self.x_shape,dtype=self.x_dtype),np.zeros(self.y_shape,dtype=self.y_dtype)
        else:
            retval_x = self.train_x[self.index:self.index+size]
            retval_y = self.train_y[self.index:self.index+size]       
            self.index = self.index + size
            return retval_x,retval_y

class File_fetcher:
    def __init__(self,prefix,max_id):
        self.prefix = prefix
        self.max_id = max_id
        self.cur_id = -1 
    
    def fetch(self):
        if self.cur_id +1 < self.max_id:
            self.cur_id = self.cur_id = 1
            bin_file = open(self.prefix+str(self.cur_id),mode='rb')
            train_data = pickle.load(bin_file)
            bin_file.close()
            return train_data
        else:
            print("Out of bounds")
            return []
    def fetch_index(self,index):
            bin_file = open(self.prefix+str(index),mode='rb')
            train_data = pickle.load(bin_file)
            bin_file.close()
            return train_data
    
    def set_lastcount(self,id_num):
        self.cur_id = id_num
    
    
class TrainingData_PQR:
    def __init__(self,path,limit=-1):
        p,q,r = gen_pqr_tuples(path,limit)
        self.p = np.asarray(p,dtype=np.float32).reshape((-1,832))
        self.q = np.asarray(q,dtype=np.float32).reshape((-1,832))
        self.r = np.asarray(r,dtype=np.float32).reshape((-1,832))       
        self.index  = 0
        self.len = len(self.p)
        if len(self.p) != len(self.q) or len(self.q) != len(self.r):
            raise Exception("p,q and r are not same length")
    
    def next_batch(self,size):
        if self.index + size >= self.len:
            print("size is larger than the remaining batch")
            return []
        else:
            retval_p = self.p[self.index:self.index+size]
            retval_q = self.q[self.index:self.index+size]           
            retval_r = self.r[self.index:self.index+size]       
            return retval_p,retval_q,retval_r        


weights_1 = new_weights(shape=[input_size,input_size])
biases_1  = new_weights(shape=[batch_size,input_size])

weights_2 = new_weights(shape=[input_size,input_size])
biases_2  = new_weights(shape=[batch_size,input_size])

weights_3 = new_weights(shape=[input_size,last_connected_output])
biases_3  = new_weights(shape=[batch_size,last_connected_output])

weights_4 = new_weights(shape=[last_connected_output,1])
biases_4  = new_weights(shape=[batch_size])

weights = [weights_1,weights_2,weights_3,weights_4]
biases  = [biases_1,biases_2,biases_3,biases_4]

def matmul_input(input, weights,biases,input_size,last_connected_output,use_RELU=False):
    retval,features = flatten_layer(input)
    for i in range(4):
        retval = tf.matmul(retval,weights[i])
        retval = retval + biases[i]
        if use_RELU:
            retval = tf.nn.relu(retval)

    return retval
        
p = tf.placeholder(tf.float32,[None,input_size])
q = tf.placeholder(tf.float32,[None,input_size])
r = tf.placeholder(tf.float32,[None,input_size])

p_input = tf.reshape(p,[-1,input_height,input_width,1])
q_input = tf.reshape(q,[-1,input_height,input_width,1])
r_input = tf.reshape(r,[-1,input_height,input_width,1])

p_val = matmul_input(p_input,weights,biases,input_size,last_connected_output)  
q_val = matmul_input(q_input,weights,biases,input_size,last_connected_output)        
r_val = matmul_input(r_input,weights,biases,input_size,last_connected_output)        


neg_likelihood = 10 * tf.square(p_val-q_val) + q_val# tf.sigmoid(p_val) + tf.sigmoid(p_val) + tf.square(p_val-q_val)#10*tf.sigmoid(q_val- p_val)* tf.sigmoid(tf.square(q_val)*tf.square( p_val))# - p_val + q_val#(q_val-p_val) + k*tf.square(q_val + p_val) + r_priority*tf.square(q_val - r_val*r_worse_than_q) 


reduced_neg_likelihood = tf.reduce_sum(neg_likelihood)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(reduced_neg_likelihood)
