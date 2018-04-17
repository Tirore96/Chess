from colorama import Fore
import numpy as np
import boardlib
import copy
import sys
import pdb
import training
import tensorflow as tf
all_squares = [[i,j] for i in range(8) for j in range(8)]
all_moves = [i+j for i in all_squares for j in all_squares]
pieces = ["Empty","Rook","Knight","Bishop","Queen","King","Pawn","Pawn","King","Queen","Bishop","Knight","Rook"]

piece_owner = [0,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1]
list_letters = ['a','b','c','d','e','f','g','h']
list_numbers = list(map(str,list((range(1,9)))))
legal_outputs = ["pawn_move","pawn_move,en-passant","king_move",'king_move,castling,kingside','king_move,castling,queenside','en-passant','']
en_passant = ["c7c5","a2a3","c5c4","b2b4","c4b3"]
zero_line = [0 for _ in range(8)]
move_zero_arr = [0 for i in range(8*8)]
flattened_zero_arr = np.zeros((8,8))
flattened_board = np.zeros((8,8,7),dtype=int)
rights_zero_arr = np.zeros((8,8,2),dtype=int)
true_board_to_my_board_dict = {'.':0,'r': 1,'n': 2,'b': 3,'q': 4,'k': 5,'p' : 6,'R': -1,'N': -2,'B': -3,'Q': -4,'K': -5, 'P': -6 }

def curate_pgn_string(s):
    f = ["\n","+","Q","N","-","R","x","B","K","#"]
    for i in f:
        s = s.replace(i,"")
    s_list = s.split(' ')
    try:
        s_list.remove("")
    except:
        s_list = s_list
    return s_list
#path = "../extracted/two.txt"
#f = open(path,"r")
#s = f.read()
#curated = curate_pgn_string_from_file(s)


def file_to_move_lists(path,groundtruth=False):
    file = open(path,"r")
    file_string = file.read()
    file_string = file_string.lower()
    #remove plus from moves if it is moves for the groundtruth
    remove_chars = ["+","#","1-0","0-1","1/2-1/2"]
    if groundtruth:
        for i in remove_chars:
            file_string = file_string.replace(i,"")
    move_list = file_string.split("\n")
  #  results = []
  #  move_list = [i for i in move_list if i not in results]
    move_list_2d = [i.split(" ") for i in move_list]
    move_size = 4
    remove_elements = [" ",""]
    if groundtruth:
        move_size = 5
    move_list_2d = [[i for i in j if i not in remove_elements] for j in move_list_2d]
    move_list_2d = [i for i in move_list_2d if i != []]
    move_list_2d_curated = [[i[:move_size] for i in j] for j in move_list_2d]
    file.close()
    return move_list_2d_curated


#Creates a new board
def create_board(): 
    #pdb.set_trace()
    val = 0
    board = np.array([[val for j in range(8)] for i in range(8)])
    for i in range(8):
        board[1,i] = 6
        board[6,i] = -6
        
    board[0,4] = 5
    board[7,4] =  -5

    board[0,1] = 2
    board[7,1] = -2

    board[0,6] = 2
    board[7,6] = -2

    board[0,2] = 3
    board[7,2] = -3

    board[0,5] = 3 
    board[7,5] = -3

    board[0,3] = 4
    board[7,3] = -4

    board[0,4] = 5
    board[7,4] = -5
    
    board[0,0] = 1
    board[7,0] = -1 
    
    board[0,7] = 1
    board[7,7] = -1
    return [board]

def create_rights():
    rights = np.array([[0 for j in range(8)] for i in range(8)])
    rights[0,4] = 1
    rights[7,4] = -1
    
    rights[0,0] = 1
    rights[7,0] = -1
    rights[0,7] = 1
    rights[7,7] = -1
    return [rights]

def create_player_positions():
    positions_1 = []
    for i in range(8):
        positions_1.append([0,i])
        positions_1.append([1,i])
    
    positions_2 = []
    for i in range(8):
        positions_2.append([6,i])
        positions_2.append([7,i])
        
    return [[positions_1] + [positions_2]]
    
def create_chess_status():
    return [False]

def create_king_positions():
    king_positions = [[0,4],[7,4]]
    return [king_positions]

def get_from_square_player(i,j,board):
    val = board[i,j]
    return val / abs(val)


def pp_board(board):
    for i in range(8):
        print(Fore.BLACK + list_numbers[7-i],end="\t")
        for j in range(8):
            player = piece_owner[board[i,j]]
            if player == 0:
                print(Fore.BLACK + "_",end="\t")
            else: 
                piece = pieces[board[i,j]]
                if player== 1:
                    print(Fore.BLUE + piece,end="\t")
                else:
                    print(Fore.BLACK + piece,end="\t")
        print("\n")
    print("",end="\t")
    for i in list_letters:
        print(Fore.BLACK + i,end="\t")
    sys.stdout.flush()
#add alpha beta pruning



class ChessBoard:
    def __init__(self,player,path="model_final"):
        self.board = create_board()
        self.rights = create_rights()
        self.king_positions = create_king_positions()
        self.player_positions = create_player_positions()
        self.chess_status = create_chess_status()
        self.player = player
        self.move_status = ""
        self.capture_status = []
        self.model = training.Model(path)
    
    def make_move(self,move):
        self.board ,self.move_status ,self.rights ,self.player ,self.king_positions ,self.player_positions ,self.chess_status,self.capture_status = boardlib.make_move(self.board,
                                                                                                                                          move,
                                                                                                                                          self.rights,
                                                                                                                                          self.player,
                                                                                                                                          self.king_positions,
                                                                                                                                          self.player_positions,
                                                                                                                                          self.chess_status)
    def unmake_move(self):
        self.board,self.move_status,self.rights,self.player,self.king_positions,self.player_positions,self.chess_status = boardlib.unmake_move(self.board,
                                                                                                                                            self.rights,
                                                                                                                                            self.player,
                                                                                                                                            self.king_positions,
                                                                                                                                            self.player_positions,
                                                                                                                                            self.chess_status)
    
    def one_hot_encode_board(self):   
        array = self.board[-1]
        flattened_board_copy = np.zeros((8,8,13),dtype=int) 
        for i in range(8):
            for j in range(8):
                piece_index = int(abs(array[i,j] ))
                sign = np.sign(piece_index)
                offset = 0
                if sign == -1:
                    offset = 6
                flattened_board_copy[i,j,int(abs(piece_index))+offset] = sign
        
        return flattened_board_copy.flatten()
    
    def negamax(self,depth,move):
        if depth==0:
            return self.evaluate_board(),move
        
        legal_moves = self.generate_legal_moves()
        score = -sys.maxsize
        move = ""
        for i in legal_moves:
            self.make_move(i)
            cur_score,cur_move = self.negamax(depth-1,i)
            cur_score = cur_score * -1
            self.unmake_move()
            if cur_score > score:
                score = cur_score
                move = cur_move
        return score,move

    def return_best_move(self,depth):
        return self.negamax(depth,"")
    
    def train_model(self,iterations,train_data,batch_size):
        self.model.run_session(iterations,train_data,batch_size)
    
    
    def show_weights(self):
        val = self.model.session.run(weights_1)
        print(val)
        
    def restore_model(self):
        self.model.restore_model()
        
    def restore_default(self):
        tf.reset_default_graph()
        
    def close_session(self):
        self.model.session.close()



    def evaluate_board(self):
        return self.model.evaluate(self)
        
    def show_board(self):
        pp_board(self.board[-1])
    
    def generate_legal_moves(self):
        return boardlib.generate_all_legal_moves(self.board,self.rights,self.player,
                                        self.king_positions,self.player_positions,
                                        self.chess_status,boardlib.all_moves)
        
