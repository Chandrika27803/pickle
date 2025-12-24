import tkinter as tk
import random
import json
import os
 
# (Old) File for player patterns – left in case you want to use later
PLAYER_HISTORY_FILE = "player_patterns.json"
 
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
 
# ---- NEW: AI "memory" file for simple ML ----
AI_MEMORY_FILE = "ai_state_values.json"
 
def load_ai_state_values():
    """Load learned state values from disk."""
    if os.path.exists(AI_MEMORY_FILE):
        try:
            with open(AI_MEMORY_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}
 
def save_ai_state_values():
    """Save learned state values to disk."""
    with open(AI_MEMORY_FILE, "w") as f:
        json.dump(ai_state_values, f)
 
def board_to_string(b):
    """Flatten board into a 9-character string."""
    return ''.join(b[i][j] for i in range(3) for j in range(3))
 
def update_ai_memory(result):
    """
    Update value estimates for board states seen this game.
    result: +1 (AI win), -1 (AI loss), 0 (draw)
    """
    global ai_state_values, ai_states_this_game
    for state in ai_states_this_game:
        info = ai_state_values.get(state, {"value": 0.0, "count": 0})
        count = info["count"] + 1
        value = info["value"]
        # Incremental average: new_value = old + (r - old)/count
        new_value = value + (result - value) / count
        ai_state_values[state] = {"value": new_value, "count": count}
    save_ai_state_values()
    ai_states_this_game = []
 
# Initialize scores and state
player_score = 0
ai_score = 0
current_player = None
 
# Load AI memory and per-game state list
ai_state_values = load_ai_state_values()
ai_states_this_game = []
 
# Create the main game window
window = tk.Tk()
window.title("Tic Tac Toe: Player vs AI (with simple ML)")
 
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
def check_winner(b, mark):
    for i in range(3):
        if all([cell == mark for cell in b[i]]):  # rows
            return True
        if all([b[j][i] == mark for j in range(3)]):  # columns
            return True
    # diagonals
    if all([b[i][i] == mark for i in range(3)]) or all([b[i][2 - i] == mark for i in range(3)]):
        return True
    return False
 
# Get available moves
def available_moves(b):
    return [(i, j) for i in range(3) for j in range(3) if b[i][j] == '-']
 
# AI strategy: Minimax (base) – no learning inside this
def minimax(b, is_ai_turn):
    if check_winner(b, 'O'):
        return 10
    if check_winner(b, 'X'):
        return -10
    if not available_moves(b):
        return 0
 
    if is_ai_turn:
        best_score = -float('inf')
        for move in available_moves(b):
            i, j = move
            b[i][j] = 'O'
            score = minimax(b, False)
            b[i][j] = '-'
            if score > best_score:
                best_score = score
        return best_score
    else:
        best_score = float('inf')
        for move in available_moves(b):
            i, j = move
            b[i][j] = 'X'
            score = minimax(b, True)
            b[i][j] = '-'
            if score < best_score:
                best_score = score
        return best_score
 
# AI move chooser: minimax + learned values
def best_move():
    """Choose AI move using minimax + simple learned state values."""
    best_score = -float('inf')
    move = None
    learn_weight = 2.0  # how much learned value influences decision
 
    for i, j in available_moves(board):
        # Try move
        board[i][j] = 'O'
        base_score = minimax(board, False)
        state_str = board_to_string(board)
        learned_info = ai_state_values.get(state_str, {"value": 0.0})
        learned_value = learned_info.get("value", 0.0)
        combined_score = base_score + learn_weight * learned_value
        # Undo move
        board[i][j] = '-'
 
        if combined_score > best_score:
            best_score = combined_score
            move = (i, j)
 
    return move
 
def end_game(winner):
    """Handle end of game: update scores, ML memory, and reset board."""
    global player_score, ai_score
 
    if winner == 'player':
        status_label.config(text="Player wins!")
        player_score += 1
        result = -1  # bad for AI
    elif winner == 'ai':
        status_label.config(text="AI wins!")
        ai_score += 1
        result = 1   # good for AI
    else:
        status_label.config(text="It's a tie!")
        result = 0   # neutral
 
    update_scores()
    update_ai_memory(result)
    reset_board()
 
# Handle player's move
def handle_player_move(row, col):
    global current_player
 
    if board[row][col] == '-' and current_player == 'Player':
        board[row][col] = 'X'
        buttons[row][col].config(text='X', state='disabled')
 
        if check_winner(board, 'X'):
            end_game('player')
            return
 
        if not available_moves(board):
            end_game('tie')
            return
 
        # Switch to AI's turn
        current_player = 'AI'
        status_label.config(text="AI's Turn")
        window.after(200, ai_move)  # small delay so UI feels responsive
 
# AI's move
def ai_move():
    global current_player
 
    ai_move_pos = best_move()
    if ai_move_pos:
        ai_row, ai_col = ai_move_pos
        board[ai_row][ai_col] = 'O'
        buttons[ai_row][ai_col].config(text='O', state='disabled')
 
        # Record state after AI move for learning
        ai_states_this_game.append(board_to_string(board))
 
        if check_winner(board, 'O'):
            end_game('ai')
            return
 
    if not available_moves(board):
        end_game('tie')
    else:
        current_player = 'Player'
        status_label.config(text="Player's Turn")
 
# Randomly select who plays first
def start_game():
    global current_player, ai_states_this_game
    reset_board()
    ai_states_this_game = []
    current_player = random.choice(['Player', 'AI'])
    status_label.config(text=f"{current_player} goes first!")
 
    if current_player == 'AI':
        window.after(200, ai_move)
 
# Reset the game board
def reset_board():
    global board, current_player, ai_states_this_game
    board = [['-' for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            buttons[i][j].config(text='', state='normal')
    current_player = None
    ai_states_this_game = []
 
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
