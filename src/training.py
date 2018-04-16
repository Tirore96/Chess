import tensorflow as tf
import sys
import boardclass
import random
import time
import numpy as np
#Credit to Hvass Laboratories
input_height = 32
input_width = 26
input_size = input_height * input_width
num_squares = 8 *8 
#x = tf.placeholder(tf.float32,[None,input_size])

#y_true_cls = tf.argmax(y_true,axis=2)
filter_size1 = 2
num_filters1 = 3

filter_size2 = 2
num_filters2 = 6
last_connected_output = 2048

fc_size = 128
learning_rate = 1e-5
k = 100
global_sess = None
batch_size = 1

class Model:
    def __init__(self,path="/tmp/model.ckpt"):
        self.session = tf.Session()
        self.path = path
    
    def run_session(self,iterations,train_data,batch_size):
        self.session.run(tf.global_variables_initializer())
        optimize(iterations,train_data,batch_size,self.session)
        saver = tf.train.Saver()
        path = saver.save(self.session,self.path)
        print("Model saved at {}".format(path))
        self.session.close()


    def restore_model(self):
        saver = tf.train.Saver()
        self.session = tf.Session()
        saver.restore(self.session,self.path)    
        self.session.run(tf.global_variables_initializer())


    def evaluate(self,board):
        p_cur = board.one_hot_encode_board()
        p_cur = p_cur.reshape((1,832))
        score = self.session.run(p_val,feed_dict={p:p_cur})
        return score
    
    def close_session(self):
        self.session.close()

#credit to Hvass Laboratories
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.5,shape=[length]))


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_falt = tf.reshape(layer,[-1,num_features])
    return layer_falt, num_features

def gen_pqr_tuples(path,limit=-1):
    p = []
    q = []
    r = []

    move_list_2d_curated = boardclass.file_to_move_lists(path,groundtruth=True)
    curated_len = len(move_list_2d_curated)
    if limit == -1:
        limit = curated_len
        
    counter =0
    for i in range(limit):
        board = boardclass.ChessBoard(-1)
        for a_move in move_list_2d_curated[i]:
 #           print(a_move)
            p_board_encoded = board.one_hot_encode_board()
            r_board_encoded = encode_board_after_random_move(board)
            board.make_move(a_move)
            q_board_encoded = board.one_hot_encode_board()
            p.append(p_board_encoded)
            q.append(q_board_encoded)
            r.append(r_board_encoded)
            #print(board.move_status,counter)
            if board.move_status == "did not make move":
                print("ERROR did not make move in gen_pqr_tupes")
                return a_move_list
            
        counter = counter + 1
        if counter % 50 == 0:
            print("{} out of {}".format(counter,limit))#str(counter) + " out of " + str(curated_len))
            
    return p,q,r
            
            
def encode_board_after_random_move(b):
    moves = b.generate_legal_moves()
    moves_len = len(moves)
    index = random.randint(0,moves_len-1)
    b.make_move(moves[index])
    one_hot = b.one_hot_encode_board()
    b.unmake_move()
    return one_hot
        

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
        
class TrainingData_PQR:
    def __init__(self,path,limit=-1):
        p,q,r = gen_pqr_tuples(path,limit)
        self.p = np.asarray(p,dtype=np.float32).reshape((-1,832))
        self.q = np.asarray(q,dtype=np.float32).reshape((-1,832))
        self.r = np.asarray(r,dtype=np.float32).reshape((-1,832))       
#        self.x_shape = self.train_x.shape
#        self.y_shape = self.train_y.shape       
#        self.x_dtype = self.train_x.dtype
#        self.y_dtype = self.train_y.dtype
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

def matmul_input(input, weights,biases,input_size,last_connected_output):
    retval,features = flatten_layer(input)
    for i in range(4):
        retval = tf.matmul(retval,weights[i])
        retval = retval + biases[i]
        retval = tf.nn.relu(retval)
 #   retval = tf.matmul(weights[4],retval)
 #   print(retval)
 #   retval = retval + biases[4]
 #   retval = tf.nn.relu(retval)
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

likelihood = tf.log(tf.sigmoid(q_val-r_val)) + k * tf.log(p_val + q_val) + k*tf.sigmoid(-q_val-p_val) #tf.sigmoid(tf.log(tf.sigmoid(r_val-q_val)) - tf.log(tf.abs(p_val - q_val))*k)
reduced_likelihood = tf.reduce_sum(likelihood)
reduced_neg_likelihood = tf.subtract(tf.cast(1,tf.float32),reduced_likelihood)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(reduced_neg_likelihood)



def optimize(iterations,train_data,batch_size,session):
    start_time = time.time()

    for i in range(iterations):
        p_batch,q_batch,r_batch = train_data.next_batch(batch_size)
        feed_dict_train = {p:p_batch,q:q_batch,r:r_batch}
        
        session.run(optimizer,feed_dict=feed_dict_train)
        
        #score = session.run(p_val_4,feed_dict={p_input:p_batch})
#        
        acc = session.run(reduced_likelihood,feed_dict=feed_dict_train)*100
        print("Optimization Iteration {}, Training Accuracy {}".format(i,acc))#+ str(i) + " Training Accuracy "+str(acc)+"" + str(score))
    end_time = time.time()
    print("Time used " + str(end_time-start_time))


def run_session(iterations,train_data,batch_size):
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    optimize(iterations,train_data,batch_size,session)
    saver = tf.train.Saver()
    path = saver.save(session,"/tmp/model.ckpt")
    print("Model saved at {}".format(path))
    session.close()
    return 


#training_size = len(train_x)
#iterations = 100
#batch_size = training_size//iterations
#
#optimize(iterations,train_x,train_y,batch_size)
