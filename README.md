# Chess
The purpose of this project, is to build a chess engine and see how well it does. The project is comprised of:

# training.py 
Here you find the tensorflow model and you can tweak parameters such as learningrate and other constants.

# test.py
This contains functions that have been of great help to me, in debugging movegeneration and countless other bugs

# Chess.ipynb
This is a python-notebook. It doesn't contain much. I use it to call the functions defined in the other python-files.

# boardlib.pyx
This contains all functions that have to do with movegeneration. This is .pyx file instead of .py, because I use Cython to compile this file to c-code. This file has sadly become very messy and unreadable. To ease the use of these functions, I added...

# boardclass.py
This encapsulates the functions in boardlib.pyx into a ChessBoard class.


# How to run it?
After cloning this repository, open Chess.ipynb. Run all cells. In the output of the last cell, a board is printed. You are black, and you enter a move by using algebraic notation. For example you can move your left-most pawn by two squares with a2a4. Finish by pressing enter. To signal the chess engine to make a move, press enter again.
