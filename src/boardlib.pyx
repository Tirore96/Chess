import numpy as np
import itertools
cimport numpy as np_c
import pdb
import copy
import sys

all_squares = [[i,j] for i in range(8) for j in range(8)]
all_moves = [i+j for i in all_squares for j in all_squares]
zero_arr = np.array([[0 for i in range(8)] for j in range(8)])
one_arr = [[1 for i in range(8)] for j in range(8)]
pieces = ["Empty","Rook","Knight","Bishop","Queen","King","Pawn","Pawn","King","Queen","Bishop","Knight","Rook"]
promotions = {'r':1,'n':2,'b':3,'q':4}
piece_owner = [0,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1]
zero_line = [0 for _ in range(8)]
piece_owner = [0,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1]
list_letters = ['a','b','c','d','e','f','g','h']
list_numbers = list(map(str,list((range(1,9)))))
slider     = [False,True,False,True,True,False, False, False, False, True, True, False, True]
x_algebraic = {"8": 0,"7": 1,"6": 2,"5": 3,"4": 4,"3": 5,"2": 6,"1": 7}
y_algebraic = {"a": 0,"b": 1,"c": 2,"d": 3,"e": 4,"f": 5,"g": 6,"h": 7}
algebraic_vertical = ['8','7','6','5','4','3','2','1']
algebraic_horisontal = ['a','b','c','d','e','f','g','h']

def make_direction_list():
    north = [-1,0]
    south = [1,0]
    east =  [0,1] 
    west= [0,-1]
    northwest= [-1,-1] 
    northeast= [-1,1] 
    southwest = [1,-1] 
    southeast= [1,1] 
    cross_dir = [north, south, east,west]
    lateral_dir = [northeast,northwest,southeast,southwest]
    all_dir = cross_dir + lateral_dir
    all_dir.extend(lateral_dir)
    NNW = np.add(np.add(north,north),west)
    NWW = np.add(np.add(north,west),west)
    NNE = np.add(np.add(north,north),east)
    NEE = np.add(np.add(north,east),east)
    SSW = np.add(np.add(south,south),west)
    SWW = np.add(np.add(south,west),west)
    SSE = np.add(np.add(south,south),east)
    SEE = np.add(np.add(south,east),east)
    knight_dir = [NNW,NWW,NNE,NEE,SSW,SWW,SSE,SEE]
    knight_dir = [i.tolist() for i in knight_dir ]
    direction_list = [[],cross_dir,knight_dir,lateral_dir,all_dir,all_dir,  all_dir,all_dir,lateral_dir,knight_dir,cross_dir]
    return direction_list

direction_list = make_direction_list()

#Used for debuggning
#def check_if_squares_attackable(board, player,sensitive_squares,
#                                    rights,player_positions):
#    enemy_index = 1 if player == 1 else 0
#    enemy_positions = player_positions[enemy_index]
#    enemy_len = len(enemy_positions)
#    sensitive_len = len(sensitive_squares)
#    #multi thread?
#    for i in range(enemy_len):
#        for j in range(sensitive_len):
#            move = enemy_positions[i] + sensitive_squares[j]
#            legal,status,_ = eval_pseudo_legal_move(board,move,rights,-player,player_positions)
#            if legal:
#                return True
#    
#    return False

cpdef check_if_squares_attackable(np_c.ndarray[long,ndim=2] board, long player,list sensitive_squares,
                                   np_c.ndarray[long,ndim=2] rights,list player_positions):
    cpdef long enemy_index = 1 if player == 1 else 0
    cpdef list enemy_positions = player_positions[enemy_index]
    cpdef int i,j,legal
    cpdef int enemy_len = len(enemy_positions)
    cpdef int sensitive_len = len(sensitive_squares)
    cpdef list move
    cpdef str status
    for i in range(enemy_len):
        for j in range(sensitive_len):
            move = enemy_positions[i] + sensitive_squares[j]
            legal,status,_ = eval_pseudo_legal_move(board,move,rights,-player,player_positions)
            if legal:
                return True

    return False


def eval_knight_move(board,direction,move,player,rights):
        x_1,y_1,x_2,y_2 = move
        player_at_square = piece_owner[board[x_2,y_2]]
        index = 2
        if not direction in direction_list[index]:
            return False,"Not legal move for knight",rights
        elif player_at_square == player:
            
            return False,"attacking own player",rights
        else:
            return True,"",rights
        
def check_if_blocked(board,move,dir_reduced):
    x_1,y_1,x_2,y_2 = move 
    x_dir,y_dir = dir_reduced
    while x_1+x_dir != x_2 or y_1+y_dir != y_2:
        x_1,y_1 = x_1+x_dir,y_1+y_dir
        player_at_square = piece_owner[board[x_1,y_1]]
        if player_at_square != 0:
            return True
    return False
        
def eval_pawn_move(board,move,direction,player,rights):
    x_1,y_1,x_2,y_2 = move
    x_d,y_d = direction
    rights[x_1,y_1] = 0
    player_at_square =piece_owner[board[x_2,y_2]]

    #check direction
    if x_d == 0:
        return False,"no vertical move",rights

    if np.sign(player)!= np.sign(x_d):
        return False,"pawn is moving the wrong direction",rights
    #if going straight
    if y_d == 0:
        #if single step
        if player_at_square != 0:
            return False, "blocked",rights
        
        if abs(x_d) == 1:
            return True,"pawn_move",rights
        #if double step
        elif abs(x_d) == 2:       
            #if pawn is at start position
            if (player == 1 and x_1 == 1) or (player == -1 and x_1 == 6):
                if board[x_1+player,y_1] == 0:
                    rights[x_2,y_2] = player
                    return True,"",rights
                else:
                    return False,"blocked",rights
            else:
                return False,"pawn tried to make double move, when not at start position",rights
        elif abs(x_d) > 2:
            return False,"pawn made to large a move",rights
    #if going lateral
    elif abs(y_d) == 1 and abs(x_d) == 1:
        #if landing square has enemy
        player_at_square = piece_owner[board[x_2,y_2]]# get_from_square_player(x_2,y_2,board)
        if player_at_square == -player:
            return True,"",rights
        elif player_at_square == 0:
            #check en-passant
            #if at fifth rank
            if (x_1 == 4 and player == 1) or (x_1 == 3 and player == -1):
                piece = pieces[board[x_2-player,y_2]]

                if piece == "Pawn" and rights[x_2-player,y_2] == -player:
                    return True, "pawn_move,en-passant",rights
                else:
                    return False,"en-passant fail, either not pawn or don't have rights",rights
            else:
                return False,"en-passant fail, not at fifth rank",rights
        else:
            return False,"made cross-move without attacking",rights
    else:
        return False,"pawn has too large horizontal movement",rights
    
    
def eval_castling(board,move,direction,rights,player,player_positions):
    x_1,y_1,x_2,y_2 = move
    x_d,y_d = direction
    rook_offset = -2 if y_2 == 2 else 1
    y_plus_offset_to_large = y_2+rook_offset > 7
    if not y_plus_offset_to_large:
        has_rights = rights[x_1,y_1] == player and rights[x_2,y_2+rook_offset] == player
    else:
        has_rights = False
    if not has_rights:
        return False,"doesn't have rights to castle",rights
    is_blocked = check_if_blocked(board,move,direction)
    if is_blocked:
        return False,"blocked doing castle",rights
    # if castling queen-side
    if y_2 == 2:
        sensitive_squares = [[x_1,y_1],[x_1,2],[x_1,3]]
        status = "king_move,castling,queenside"
    # if castling king-side
    else:
        status = "king_move,castling,kingside"
        sensitive_squares = [[x_1,y_1],[x_1,5]]
       
    is_in_check = check_if_squares_attackable(board,player,sensitive_squares,rights,player_positions)
    if is_in_check:
        return False,"Is in check, trying castle",rights
    rights[x_1,y_1] = 0
    rights[x_2,y_2+rook_offset] = 0
    return True,status,rights
        
def eval_pseudo_legal_move(board,move,
                               rights,player_mover,player_positions):
    x_1,y_1,x_2,y_2 = move

    if x_2 < 0 or x_2 > 7 or y_2 < 0 or y_2 > 7:
        return False,"out of bounds",rights
    elif x_1 < 0 or x_1 > 7 or y_1 < 0 or y_1 > 7:
        return False,"out of bounds",rights
    
    if x_1 == x_2 and y_1 == y_2:
        return False,"move doesn't move piece",rights

    player = piece_owner[board[x_1,y_1]]
    
    if player != player_mover:
        return False,"player number is wrong",rights
    player_2 = piece_owner[board[x_2,y_2]]
    if player_2 == player_mover:
        return False,"friendly fire",rights

    #clear en-passant rights for player 2 turns ago:
    if player_mover == 1:
        rank_remove_rights = 3
    else:
        rank_remove_rights = 4
    rights[rank_remove_rights] = zero_line

    piece = pieces[board[x_1,y_1]]
    
    #calc difference between positions
    x_d,y_d = x_2-x_1,y_2-y_1
    
    #special case for knight
    if piece == "Knight":
        return eval_knight_move(board,[x_d,y_d],move,player,rights)

    #special case for pawn
    elif piece == "Pawn":
        return eval_pawn_move(board,move,[x_d,y_d],player,rights)
    
    #try to reduce direction to eg. [1,0]
    #if x_d and y_d don't have same absolute size and neither of them is 0, it's not a line
    if (abs(x_d) != abs(y_d)) and not(x_d == 0 or y_d ==0):
        return False,"move is not a line",rights
    x_dir = np.sign(x_d)
    y_dir = np.sign(y_d)
    
    #check direction is allowed for piece, and that it only slides (move more than one square) if it is allowed
    index = board[x_1,y_1]
    if [x_dir,y_dir] in direction_list[index]:
        blocked = check_if_blocked(board,move,[x_dir,y_dir])
        if not blocked:
            is_slider = slider[index]
            if not is_slider:
                #only king satisfies constraints this deep into if-statements
                if abs(x_d) > 1 or abs(y_d) > 1:
                    if piece == "King":
                        return eval_castling(board,move,[x_dir,y_dir],rights,player,player_positions)
                    return False,"Piece is not a slider, but tries to slide",rights
                #normal king move of one
                rights[x_1,y_1] = 0
                return True,"king_move",rights
            rights[x_1,y_1] = 0
            return True,"",rights
        return False,"piece is blocked",rights
    return False,"Unrecognized violation",rights

def algebraic_to_arr_indices(s):
    try:
        y_1,x_1,y_2,x_2 = s
    except:
        pdb.set_trace()
    x_1_alg = x_algebraic[x_1]
    y_1_alg = y_algebraic[y_1]
    x_2_alg = x_algebraic[x_2]
    y_2_alg = y_algebraic[y_2]
    move = [x_1_alg,y_1_alg,x_2_alg,y_2_alg]
    return move

def make_move(board,move,rights,player,king_pos,player_positions,chess_status,move_format="alg"):
    has_promotion = len(move) == 5
    promotion = ""
    if has_promotion:
        promotion = move[4] 
    if move_format == "alg":
        if has_promotion:
            move = move[:-1]
        move = algebraic_to_arr_indices(move)
        
    x_1_alg,y_1_alg,x_2_alg,y_2_alg = move
    cur_board = copy.deepcopy(board[-1])
    cur_rights = copy.deepcopy(rights[-1])
    cur_king_pos = copy.deepcopy(king_pos[-1])
    cur_player_positions = copy.deepcopy(player_positions[-1])

    legal_move,status,cur_rights,now_in_chess = eval_legal_move(board,move,rights,player,king_pos,player_positions,chess_status[-1],promotion)
    player_index = 0 if player == 1 else 1
    enemy_index = 1 if player == 1 else 0
    capture_status = []
    if legal_move:
        index = cur_board[x_1_alg,y_1_alg]
        piece = pieces[index]
        aux_move = []
        capture = (piece_owner[cur_board[x_2_alg,y_2_alg]] == -player) or "en-passant" in status
        if capture:
            capture_status = [str(player),pieces[cur_board[x_2_alg,y_2_alg]]]
        if piece == "Pawn":
            cur_board,aux_move = move_pawn(cur_board,move,status,player,promotion)
            
        elif piece == "King":
            cur_board,aux_move = move_king(cur_board,move,status)
            cur_king_pos = update_king_pos(cur_king_pos,player_index,x_2_alg,y_2_alg)
        else:
            cur_board[x_2_alg,y_2_alg] = cur_board[x_1_alg,y_1_alg]
            cur_board[x_1_alg,y_1_alg] = 0
            if capture:
                cur_player_positions[player_index].remove([x_1_alg,y_1_alg])#(original_move_before)
        
        if aux_move != []:
            if aux_move[2] == -1:
                #en-passant attack
                pos_remove = [aux_move[0],aux_move[1]]
            
                cur_player_positions[enemy_index].remove(pos_remove)
            else:
                pos_before = aux_move[0:2]
                pos_after = aux_move[2:4]

                cur_player_positions[player_index].remove(pos_before)
                cur_player_positions[player_index].append(pos_after)
        original_move_before = move[0:2]
        original_move_after = move[2:4]  

        cur_player_positions[player_index].append(([x_2_alg,y_2_alg]))    
        
        board.append(cur_board)
        rights.append(cur_rights)
        king_pos.append(cur_king_pos)
        player_positions.append(cur_player_positions)
        chess_status.append(now_in_chess)
        return board,True,rights,-player,king_pos,player_positions,chess_status,capture_status
    else:
        return board,False,rights,player,king_pos,player_positions,chess_status,capture_status

def update_king_pos(king_pos,player,x,y):
    king_pos[player] = [x,y]
    return king_pos



def eval_legal_move(board,move,rights,player,king_pos,player_positions,chess_status,promotion):
    cur_board = copy.copy(board[-1])
    cur_rights = copy.deepcopy(rights[-1])
    cur_king_pos = copy.copy(king_pos[-1])
    legal_move,status,rights_new = eval_pseudo_legal_move(cur_board,move,cur_rights,player,player_positions[-1])
    #if move is not legal, don't check if the king is in chess.
    if not legal_move:
        return False,status,rights_new,False
    king_is_now_in_chess = is_king_now_in_chess(board,move,rights,king_pos,cur_board,cur_rights,cur_king_pos,player,player_positions,chess_status,status,promotion)
    retval = legal_move and not king_is_now_in_chess
    return retval,status,rights_new,False

def gen_all_possible_moves(all_moves):
    players = [-1,1]
    remove = []
    for i in all_moves:
        x_1,y_1,x_2,y_2 = i
        cp = copy.deepcopy(zero_arr)
        global_keep = False
        for j in range(1,7):
            cp[x_1,y_1] = j
            for k in players:
                keep = eval_pseudo_legal_move(cp,i,
                               one_arr,k,[[[]],[[]]])
                if keep:
                    global_keep = True
                    break
        if not global_keep:
            remove.append(i)
    for i in remove:
        all_moves.remove(i)
    return all_moves
            
all_moves = gen_all_possible_moves(all_moves)

#def generate_all_legal_moves(board,rights,player,king_pos, player_positions,chess_status,all_moves):
#
#    all_legal_moves = []
#    all_moves_len = len(all_moves)
#    for i in range(all_moves_len):
#        move = all_moves[i]
#        legal,_,_,_ = eval_legal_move(board,move,rights,player,king_pos,player_positions,chess_status[-1],"")
#        if legal:
#            algebraic_move = arr_to_algebraic(move)
#            all_legal_moves.append(algebraic_move)
#    return all_legal_moves

cpdef generate_all_legal_moves(list board,list rights,long player,list king_pos, list player_positions,list chess_status,list all_moves):

    cpdef list all_legal_moves = []
    cpdef int all_moves_len = len(all_moves)
    cpdef int i
    cpdef int legal
    cpdef str algebraic_move
    cpdef list move
    for i in range(all_moves_len):
        move = all_moves[i]
        legal,_,_,_ = eval_legal_move(board,move,rights,player,king_pos,player_positions,chess_status[-1],"")
        if legal:
            algebraic_move = arr_to_algebraic(move)
            all_legal_moves.append(algebraic_move)
    return all_legal_moves

def legal_move_exists(board,rights,player,king_pos, player_positions,chess_status,all_moves):
    all_legal_moves = []
    all_moves_len = len(all_moves)
    for i in range(all_moves_len):
        move = all_moves[i]
        legal,_,_,_ = eval_legal_move(board,move,rights,player,king_pos,player_positions,chess_status[-1],"")
        if legal:
            return True
    return False 

def arr_to_algebraic(arr):
    x_1,y_1,x_2,y_2 = arr
    x_1_s =algebraic_vertical[x_1]
    y_1_s =algebraic_horisontal[y_1]
    x_2_s =algebraic_vertical[x_2]
    y_2_s =algebraic_horisontal[y_2]
    return y_1_s + x_1_s + y_2_s + x_2_s


#retrieves the previous state of the board
def unmake_move(board,rights,player,king_pos,player_positions,chess_status):
    if len(board) == 1:
        #don't change player if board hasn't changed
        return board,False,rights,player,king_pos,player_positions,chess_status

    board.pop()
    rights.pop()
    king_pos.pop()
    player_positions.pop()
    chess_status.pop()
    return board,True,rights,-player,king_pos,player_positions,chess_status


def move_king(cur_board,move,status):
    x_1_alg,y_1_alg,x_2_alg,y_2_alg = move
    aux_move = []
    if "castling" in status:
        side = status.split(',')
        rook_offset_from = -2 if side[2] == 'queenside' else 1
        rook_offset_to = 1 if side[2] == 'queenside' else -1
        
        #moving the rook
        cur_board[x_2_alg,y_2_alg+rook_offset_to] = cur_board[x_2_alg,y_2_alg+rook_offset_from]
        cur_board[x_2_alg,y_2_alg+rook_offset_from] = 0
        aux_move = [x_2_alg,y_2_alg+rook_offset_from,x_2_alg,y_2_alg+rook_offset_to]
        
    # move king no matter what
    cur_board[x_2_alg,y_2_alg] = cur_board[x_1_alg,y_1_alg]
    cur_board[x_1_alg,y_1_alg] = 0
    return cur_board,aux_move
    
def move_pawn(cur_board,move,status,player,promotion):
    x_1_alg,y_1_alg,x_2_alg,y_2_alg = move
    aux_move = []
    promotable_position = x_2_alg == 7 if player == 1 else x_2_alg == 0
    #remove enemy from enpassant
    if "en-passant" in status:
        cur_board[x_2_alg-player,y_2_alg] = 0
        aux_move = [x_2_alg-player,y_2_alg,-1,-1]

    if promotable_position:
        if promotion == "":
            promotion = "q"
        piece = promotions[promotion] * player
    else:
        piece = cur_board[x_1_alg,y_1_alg]
    cur_board[x_2_alg,y_2_alg] = piece
    cur_board[x_1_alg,y_1_alg] = 0 
    return cur_board,aux_move

def is_king_now_in_chess(board,move,rights,king_pos,cur_board,cur_rights,cur_king_pos,player,player_positions,chess_status,status,promotion):
    x_1,y_1,x_2,y_2 = move
    king_index = 0 if player == 1 else 1
        
    #make move. player positions don't have to be updated, since only the enemies positions are relevant, which won't change on the players turn.
    piece = pieces[cur_board[x_1,y_1]]
    player_index = 0 if player == 1 else 1

    if piece == "Pawn":
        cur_board,_= move_pawn(cur_board,move,status,player,promotion)
        
    elif piece == "King":
        cur_board,_= move_king(cur_board,move,status)
        cur_king_pos = update_king_pos(cur_king_pos,player_index,x_2,y_2)
    else:
        cur_board[x_2,y_2] = cur_board[x_1,y_1]
        cur_board[x_1,y_1] = 0   

    king_arr = [cur_king_pos[king_index]]
    king_in_chess_now = check_if_squares_attackable(cur_board,player,king_arr,cur_rights,player_positions[-1])
    return king_in_chess_now