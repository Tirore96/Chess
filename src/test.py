## Test cell. Here are the functions that test the engine
import boardclass
import chess 

def perft(b,depth):
    if depth == 0:
        return 1
    nodes = 0
    moves = b.generate_legal_moves()
    
    for i in moves:
        b.make_move(i)
        nodes = nodes + perft(b,depth-1)
        b.unmake_move()
    return nodes

def perft_pool(b,depth):
    if depth == 0:
        return 1
    nodes = 0
    moves = b.generate_legal_moves()
    board_list = []
    for i in moves:
        b.make_move(i)
        board_list.append((copy.deepcopy(b),depth-1))
        #nodes = nodes + perft(b,depth-1)
        b.unmake_move()
    p = Pool()
    retvals = p.starmap_async(perft,board_list)
    #print(retvals.get())
    return sum(retvals.get()) 

def perft_test(depth):
    b = boardclass.ChessBoard(-1)
    start = time.time()
    val = perft_pool(b,depth)
    end = time.time() 
    return val,end-start

#If perft score is wrong. Use this function to see which moves that differ from the ground truth.
def wrong_moves(correct_board,my_board, depth,path):
    if depth== 0:
        return 1

    nodes = 0
    #pdb.set_trace()
    correct_legal_moves = list(map(correct_board.lan,correct_board.legal_moves))
    my_legal_moves = my_board.generate_legal_moves()
    concated_moves = ""
    for i in correct_legal_moves:
        concated_moves = concated_moves + i + ' '
    correct_legal_moves = curate_pgn_string(concated_moves)
    
    false_positive = ["-" + i for i in my_legal_moves if i not in correct_legal_moves]
    false_negative = ["+" + i for i in correct_legal_moves if i not in my_legal_moves]
    
    correct_legal_moves.sort()
    my_legal_moves.sort()
    if correct_legal_moves != my_legal_moves:
        print("error")
    errors = len(false_positive)+ len(false_negative)
    if errors == 0:
        for i in my_legal_moves:
            path_temp = path + " " + i
            move = chess.Move.from_uci(i)
            
            correct_board.push(move)
            my_board.make_move(i)
            
            nodes = nodes + wrong_moves(correct_board,my_board,depth-1,path_temp)
            
            correct_board.pop()
            my_board.unmake_move()
    else:
        print(path + "\n" + str(false_positive) + str(false_negative) + "\n")
        return 1
    return nodes

def test_one_hot_encoding():
    b = boardclass.ChessBoard(-1)
    b_board = b.board[-1]
    b_rights = b.rights[-1]
    one_hot = b.one_hot_encode_board()
    for i in range(8):
        if i not in [0,1,6,7]:
            continue
        for j in range(8):
            index = i * 8*13  + j *13 
            offset = 0
            if index < 0:
                offset = 6
            index = index + int(abs(b_board[i,j]))
            print(one_hot[index],int(abs(b_board[i,j]))+offset)

            
def start_game(player,moves=[],watching=False,fast_play=False,fast_play_count=0,clear=True,set_pdb=False):
    #pdb.set_trace()
    b = boardclass.ChessBoard(player)
    if fast_play == True:
        if fast_play_count == 0:
            length = len(moves)
        else:
            length = fast_play_count
        for i in range(length):
            b.make_move(moves[i])
        i = lengh
    else:
        i = 0
        
    count = 0
    do = "m"
    prev = "m"
    stop = False
    b.show_board()
    s = ""
    while not stop:
        if set_pdb:
            pdb.set_trace()
        do = input()
#        if do == "":
#            do = prev
#            
        if watching:
            if i == len(moves):
                stop = True
                
            if do == "m":
                b.make_move(moves[i])
                prev == "m"
                
                print(moves[i])
            
            elif do == "u":
                b.unmake_move()
                if s == "n":
                    continue
                i = i-2
                prev = "u"
                
                print(moves[i])
            elif do == "exit":
                stop = True
            else:
                print("wrong input")
                continue
        else:  
            if do == "u":
                b.unmake_move()
            
            else:
                b.make_move(do)
                #pdb.set_trace()
                s = b.move_status
                if not(s in legal_outputs):
                    print(s)
                    print(len(s))
                    print("wrong input")
                    sys.stdout.flush()


                    continue

        if clear:
            clear_output()
        b.show_board()
        print(b.move_status)
        sys.stdout.flush()
        count += 1
        i = i + 1
    #clear_output()
    b.show_board()

def compare_boards_with_moves(moves,moves_long):
    board_mine = boardclass.ChessBoard(-1)
    lookback = 10
    board_true = chess.Board()
    length_moves = len(moves)
    for i in range(length_moves):
        board_mine.make_move(moves[i])
        
        board_true_move = chess.Move.from_uci(moves_long[i])
        board_true.push(board_true_move)
        is_equal,status = compare_boards(board_mine.board[-1],board_true)
        if not is_equal:
            status = status + "Previous moves: {0}, reached {1} out of {2}. Move is {3}".format(str(moves[i-lookback:i]),i,length_moves,moves_long[i])
            return False,status
    return True,""

def compare_boards(board_mine,board_true):
    for i in range(8):
        for j in range(8):
            square_mine = board_mine[i,j]
            index_true = index_into_python_board(board_true,i,j)
            square_true = boardclass.true_board_to_my_board_dict[index_true]
            if square_mine != square_true:
                status = "At {0}, {1}, expected {2}, got {3}. ".format(str(i),str(j),str(square_true),str(square_mine))
     #           status = "At " + str(i) + ", " + str(j) + ", expected" + str(square_true) + ", got " + str(square_mine)
                return False,status
    return True,""
    
def index_into_python_board(board,i,j):
    str_board = str(board)
    index = i * 16 + j * 2
    return str_board[index]

def compare_boards_with_file(path):
    moves_lists = boardclass.file_to_move_lists(path)
    moves_lists_long = boardclass.file_to_move_lists(path,groundtruth=True)   
    moves_length = len(moves_lists)
    for i in range(moves_length):
        retval,status = compare_boards_with_moves(moves_lists[i],moves_lists_long[i])
        if not retval:
            print(status)
            break
    return
    
