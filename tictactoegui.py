import tkinter as tk
import random
import json
import os
 
# File to store player patterns
PLAYER_HISTORY_FILE = "player_patterns.json"
 
# Load and save player history patterns
def load_player_patterns():
 	if os.path.exists(PLAYER_HISTORY_FILE):
     	with open(PLAYER_HISTORY_FILE, "r") as file:
         	return json.load(file)
 	return []
 
def save_player_pattern(pattern):
 	history = load_player_patterns()
 	history.append(pattern)
 	with open(PLAYER_HISTORY_FILE, "w") as file:
     	json.dump(history, file)
 
# Initialize scores
player_score = 0
ai_score = 0
current_player = None
 
# Create the main game window
window = tk.Tk()
window.title("Tic Tac Toe: Player vs AI")
 
# Create the board and score labels
board = [['-' for _ in range(3)] for _ in range(3)]
buttons = [[None for _ in range(3)] for _ in range(3)]
status_label = tk.Label(window, text="Click 'Start Game' to begin", font=('Arial', 14))
status_label.pack()
 
player_score_label = tk.Label(window, text="Player: 0", font=('Arial', 12))
player_score_label.pack()
 
ai_score_label = tk.Label(window, text="AI: 0", font=('Arial', 12))
ai_score_label.pack()
 
# Check if the player or AI has won
def check_winner(board, mark):
 	for i in range(3):
     	if all([cell == mark for cell in board[i]]):  # Check rows
         	return True
     	if all([board[j][i] == mark for j in range(3)]):  # Check columns
         	return True
 	if all([board[i][i] == mark for i in range(3)]) or all([board[i][2 - i] == mark for i in range(3)]):  # Diagonals
     	return True
 	return False
 
# Get available moves
def available_moves(board):
 	return [(i, j) for i in range(3) for j in range(3) if board[i][j] == '-']
 
# AI strategy: Minimax with pattern-based learning
def minimax(board, is_ai_turn):
 	if check_winner(board, 'O'):
     	return 10
 	if check_winner(board, 'X'):
     	return -10
 	if not available_moves(board):
     	return 0
 
	if is_ai_turn:
     	best_score = -float('inf')
     	for move in available_moves(board):
         	i, j = move
         	board[i][j] = 'O'
         	score = minimax(board, False)
         	board[i][j] = '-'
         	best_score = max(best_score, score)
     	return best_score
 	else:
     	best_score = float('inf')
     	for move in available_moves(board):
         	i, j = move
         	board[i][j] = 'X'
         	score = minimax(board, True)
         	board[i][j] = '-'
         	best_score = min(best_score, score)
     	return best_score
 
# Make the best AI move
def best_move():
 	best_score = -float('inf')
 	move = None
 	for i, j in available_moves(board):
     	board[i][j] = 'O'
     	score = minimax(board, False)
     	board[i][j] = '-'
     	if score > best_score:
         	best_score = score
         	move = (i, j)
 	return move
 
# Handle player's move
def handle_player_move(row, col):
 	global player_score, ai_score, current_player
 
	if board[row][col] == '-' and current_player == 'Player':
     	board[row][col] = 'X'
     	buttons[row][col].config(text='X', state='disabled')
 
    	if check_winner(board, 'X'):
         	status_label.config(text="Player wins!")
         	player_score += 1
         	update_scores()
         	reset_board()
         	return
 
    	if not available_moves(board):
         	status_label.config(text="It's a tie!")
         	reset_board()
         	return
 
    	# Switch to AI's turn
     	current_player = 'AI'
     	ai_move()
 
# AI's move
def ai_move():
 	global current_player, ai_score
 
	ai_move = best_move()
 	if ai_move:
     	ai_row, ai_col = ai_move
     	board[ai_row][ai_col] = 'O'
     	buttons[ai_row][ai_col].config(text='O', state='disabled')
 
    	if check_winner(board, 'O'):
         	status_label.config(text="AI wins!")
         	ai_score += 1
         	update_scores()
         	reset_board()
         	return
 
	if not available_moves(board):
     	status_label.config(text="It's a tie!")
     	reset_board()
 	else:
     	current_player = 'Player'
     	status_label.config(text="Player's Turn")
 
# Randomly select who plays first
def start_game():
 	global current_player
 	reset_board()
 	current_player = random.choice(['Player', 'AI'])
 	status_label.config(text=f"{current_player} goes first!")
 
	if current_player == 'AI':
     	ai_move()
 
# Reset the game board
def reset_board():
 	global board, current_player
 	board = [['-' for _ in range(3)] for _ in range(3)]
 	for i in range(3):
     	for j in range(3):
         	buttons[i][j].config(text='', state='normal')
 	current_player = None
 
# Update score display
def update_scores():
 	player_score_label.config(text=f"Player: {player_score}")
 	ai_score_label.config(text=f"AI: {ai_score}")
 
# Create the game grid and bind buttons
frame = tk.Frame(window)
frame.pack()
 
for i in range(3):
 	for j in range(3):
     	button = tk.Button(frame, text='', font=('Arial', 20), width=5, height=2,
                        	command=lambda row=i, col=j: handle_player_move(row, col))
     	button.grid(row=i, column=j)
     	buttons[i][j] = button
 
# Start game button
start_button = tk.Button(window, text="Start Game", font=('Arial', 12), command=start_game)
start_button.pack()
 
# Reset button
reset_button = tk.Button(window, text="Reset Game", font=('Arial', 12), command=reset_board)
reset_button.pack()
 
# Run the game loop
window.mainloop()
 


