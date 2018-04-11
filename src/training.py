#Credit to Hvass Laboratories
input_height = 32
input_width = 26
input_size = input_height * input_width
num_squares = 8 *8 
#x = tf.placeholder(tf.float32,[None,input_size])
p = tf.placeholder(tf.float32,[None,input_size])
r = tf.placeholder(tf.float32,[None,input_size])
q = tf.placeholder(tf.float32,[None,input_size])
y_true = tf.placeholder(tf.float32,[None])
#y_true_cls = tf.argmax(y_true,axis=2)
filter_size1 = 2
num_filters1 = 3

filter_size2 = 2
num_filters2 = 6
last_connected_output = 2048

fc_size = 128
learning_rate = 1e-5
k = 10


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

def gen_pqr_tuples(path):
    p = []
    q = []
    r = []

    move_list_2d_curated = file_to_move_list(path)
    curated_len = len(move_list_2d_curated)
    counter = 0
    for a_move_list in move_list_2d_curated:
        board = ChessBoard(-1)
        for a_move in a_move_list:
 #           print(a_move)
            p_board_encoded = board.one_hot_encode_board()
            #r_board_encoded = encode_board_after_random_move(board)
            board.make_move(a_move)
            q_board_encoded = board.one_hot_encode_board()
            p.append(p_board_encoded)
            q.append(q_board_encoded)
            #r.append(r_board_encoded)
            #print(board.move_status,counter)
            counter = counter + 1
            if board.move_status == "did not make move":
                return a_move_list[:counter]
        if counter % 50 == 0:
            print(str(counter) + " out of " + str(curated_len))
    return []#p,q,r
            
            
def encode_board_after_random_move(b):
    moves = b.generate_legal_moves()
    p
    moves_len = len(moves)
    index = random.randint(0,moves_len-1)
    b.make_move(moves[index])
    one_hot = b.one_hot_encode_board()
    b.unmake_move()
    return one_hot
        

compare_boards_with_file("../extracted/good_games.txt")
#temp = gen_pqr_tuples("../extracted/good_games.txt")
#print(temp,len(temp))
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
        


p_input = tf.reshape(p,[-1,input_height,input_width,1])
r_input = tf.reshape(r,[-1,input_height,input_width,1])
q_input = tf.reshape(q,[-1,input_height,input_width,1])

p_flat,p_features= flatten_layer(p_input)
r_flat,r_features= flatten_layer(r_input)
q_flat,q_features= flatten_layer(q_input)


weights_1 = new_weights(shape=[input_size,input_size])
biases_1  = new_weights(shape=[input_size,input_size])

weights_2 = new_weights(shape=[input_size,input_size])
biases_2  = new_weights(shape=[input_size,input_size])

weights_3 = new_weights(shape=[input_size,last_connected_output])
biases_3  = new_weights(shape=[input_size,last_connected_output])

weights_4 = new_weights(shape=[last_connected_output,1])
biases_4  = new_weights(shape=[1])

p_val_1 = tf.matmul(p_flat,weights_1)  + biases_1
p_val_1 = tf.nn.relu(p_val_1)

p_val_2 = tf.matmul(p_val_1,weights_2) + biases_2
p_val_2 = tf.nn.relu(p_val_2)

p_val_3 = tf.matmul(p_val_2,weights_3) + biases_3
p_val_3 = tf.nn.relu(p_val_3)

p_val_4 = tf.matmul(p_val_3,weights_4) + biases_4
p_val_4 = tf.nn.relu(p_val_4)


r_val_1 = tf.matmul(r_flat,weights_1)  + biases_1
r_val_1 = tf.nn.relu(r_val_1)

r_val_2 = tf.matmul(r_val_1,weights_2) + biases_2
r_val_2 = tf.nn.relu(r_val_2)

r_val_3 = tf.matmul(r_val_2,weights_3) + biases_3
r_val_3 = tf.nn.relu(r_val_3)

r_val_4 = tf.matmul(r_val_3,weights_4) + biases_4
r_val_4 = tf.nn.relu(r_val_4)


q_val_1 = tf.matmul(q_flat,weights_1)  + biases_1
q_val_1 = tf.nn.relu(q_val_1)

q_val_2 = tf.matmul(q_val_1,weights_2) + biases_2
q_val_2 = tf.nn.relu(q_val_2)

q_val_3 = tf.matmul(q_val_2,weights_3) + biases_3
q_val_3 = tf.nn.relu(q_val_3)

q_val_4 = tf.matmul(q_val_3,weights_4) + biases_4
q_val_4 = tf.nn.relu(q_val_4)



likelihood = tf.log(tf.sigmoid(q_val_4-r_val_4)) + k * tf.log(p_val_4 + q_val_4) + k * tf.log(-p_val_4-q_val_4)
neg_likelihood = tf.subtract(tf.cast(1,tf.float32),likelihood)

#
#layer_fc2_reshaped = tf.reshape(layer_fc2,[-1,2,num_squares])
#
#y_pred = tf.nn.softmax(layer_fc2_reshaped)
#y_pred_cls = tf.argmax(y_pred,axis=2)

#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2_reshaped,
                                                 #      labels=y_true)
#cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(neg_likelihood)

#correct_prediction = tf.equal(y_pred_cls,y_true_cls)
#
#accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))



session = tf.Session()
session.run(tf.global_variables_initializer())


def optimize(iterations,train_x,train_y,batch_size):
    start_time = time.time()
    train_data = TrainingData(train_x,train_y)

    for i in range(iterations):
        x_batch,y_batch = train_data.next_batch(batch_size)
        
        feed_dict_train = {x:x_batch,y_true:y_batch}
        
        session.run(optimizer,feed_dict=feed_dict_train)
        
        acc = session.run(accuracy, feed_dict=feed_dict_train)
        print("Optimization Iteration "+ str(i) + " Training Accuracy "+str(acc))
    end_time = time.time()
    print("Time used " + str(end_time-start_time))


training_size = len(train_x)
iterations = 100
batch_size = training_size//iterations

optimize(iterations,train_x,train_y,batch_size)
