from colorama import Fore
import numpy as np
import boardlib
import importlib
importlib.reload(boardlib)
import copy
import sys
import pdb
import training
import tensorflow as tf
import chess
import chess.svg
from IPython.display import SVG

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
        print(Fore.BLACK + boardlib.list_numbers[7-i],end="\t")
        for j in range(8):
            player = boardlib.piece_owner[board[i,j]]
            if player == 0:
                print(Fore.BLACK + "_",end="\t")
            else: 
                piece = boardlib.pieces[board[i,j]]
                if player== 1:
                    print(Fore.BLUE + piece,end="\t")
                else:
                    print(Fore.BLACK + piece,end="\t")
        print("\n")
    print("",end="\t")
    for i in boardlib.list_letters:
        print(Fore.BLACK + i,end="\t")
    sys.stdout.flush()

class ChessBoard:
    def __init__(self,player,path="model_final",sync_board_on_pop=False):
        self.board = create_board()
        self.rights = create_rights()
        self.king_positions = create_king_positions()
        self.player_positions = create_player_positions()
        self.chess_status = create_chess_status()
        self.player = player
        self.made_move=None
        self.unmade_move=None
        self.capture_status = []
        self.model = training.Model(path)
        self.player_in_chess = None
        self.python_board = chess.Board()
        self.sync_board_on_pop = sync_board_on_pop
    
    def make_move(self,move,sync=True):
        if self.player_in_chess != None:
            print("Cannot make move. Player {} is in chess".format(self.player))
            pdb.set_trace()
            
        else:
            if sync:
                self.python_board.push(chess.Move.from_uci(move))
                _,_,x_2,y_2 = boardlib.algebraic_to_arr_indices(move)
                piece_color = self.player == -1
            
                #7 minus x_2 because algebraic_to_arr_indicies() assumes top-down indexing, but we want bottom-up
                square = chess.square(y_2,7-x_2)
                piece = chess.Piece(self.python_board.piece_at(square).piece_type,piece_color)
    
                self.python_board.set_piece_at(square,piece)
            
            self.board ,self.made_move,self.rights ,self.player ,self.king_positions ,self.player_positions ,self.chess_status,self.capture_status = boardlib.make_move(self.board,
                                                            move,
                                                            self.rights,
                                                            self.player,
                                                            self.king_positions,
                                                            self.player_positions,
                                                            self.chess_status)
        
            not_checkmate = boardlib.legal_move_exists(self.board,self.rights,self.player,
                                                       self.king_positions,self.player_positions,
                                                       self.chess_status,boardlib.all_moves)
            
            if not not_checkmate:
                self.player_in_chess = self.player

    def unmake_move(self):
        self.player_in_chess = None
        if self.sync_board_on_pop:
            self.python_board.pop()
        self.board,self.unmade_move,self.rights,self.player,self.king_positions,self.player_positions,self.chess_status = boardlib.unmake_move(self.board,
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
            self.make_move(i,sync=False)
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
    
    def restore_model(self):
        self.model.restore_model()
        
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