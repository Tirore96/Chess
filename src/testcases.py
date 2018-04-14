import unittest
import chess
import test
import boardclass
import boardlib
import importlib
importlib.reload(boardlib)

class TestMethods_index_into_python_board(unittest.TestCase):
    def setUp(self):
        self.board = chess.Board()
    def test_index_king_CAP(self):
        self.assertEqual(test.index_into_python_board(self.board,7,4),"K")
                         
    def test_index_king_low(self):
        self.assertEqual(test.index_into_python_board(self.board,0,4),"k")
                         
    def test_index_empty(self):
        self.assertEqual(test.index_into_python_board(self.board,2,0),".")
                         
    def test_index_rook_CAP(self):
        self.assertEqual(test.index_into_python_board(self.board,7,0),"R")

    def test_index_rook_low(self):
        self.assertEqual(test.index_into_python_board(self.board,0,0),"r")

    def test_index_knight_CAP(self):
        self.assertEqual(test.index_into_python_board(self.board,7,1),"N")

    def test_index_knight_low(self):
        self.assertEqual(test.index_into_python_board(self.board,0,1),"n")

    def test_index_queen_CAP(self):
        self.assertEqual(test.index_into_python_board(self.board,7,3),"Q")

    def test_index_queen_low(self):
        self.assertEqual(test.index_into_python_board(self.board,0,3),"q")

    def test_index_pawn_CAP(self):
        self.assertEqual(test.index_into_python_board(self.board,6,0),"P")

    def test_index_pawn_low(self):
        self.assertEqual(test.index_into_python_board(self.board,1,0),"p")
        
        
        
class Test_bug_update_rights(unittest.TestCase):
    def setUp(self):
        self.board = boardclass.ChessBoard(-1)
        
    def test_update_rights_top_left(self):
        x = 0
        y = 0
        moves = ["a2a3","a7a6","a3a4","a8a7"]
        for i in moves:
            self.board.make_move(i)
        right = self.board.rights[-1][x,y]
        self.assertEqual(right,0)
    
    def test_update_rights_top_center(self):
        x = 0
        y = 4
        moves = ["e2e3","e7e6","e3e4","e8e7"]
        for i in moves:
            self.board.make_move(i)
        right = self.board.rights[-1][x,y]
        self.assertEqual(right,0)
        
    def test_update_rights_top_right(self):
        x = 0
        y = 7
        moves = ["h2h3","h7h6","h3h4","h8h7"]
        for i in moves:
            self.board.make_move(i)
        right = self.board.rights[-1][x,y]
        self.assertEqual(right,0)
    
    def test_update_rights_bottom_left(self):
        x = 7
        y = 0
        moves = ["a2a3","a7a6","a1a2"]

        for i in moves:
            self.board.make_move(i)
        right = self.board.rights[-1][x,y]
        self.assertEqual(right,0)
    
    def test_update_rights_bottom_center(self):
        x = 7
        y = 4
        moves = ["e2e3","e7e6","e1e2"]

        for i in moves:
            self.board.make_move(i)
        right = self.board.rights[-1][x,y]
        self.assertEqual(right,0)
    
    def test_update_rights_bottom_right(self):
        x = 7
        y = 7
        moves = ["h2h3","h7h6","h1h2"]

        for i in moves:
            self.board.make_move(i)
        right = self.board.rights[-1][x,y]
        self.assertEqual(right,0)
        
        
if __name__ == '__main__':
    unittest.main()