import random
import pickle
import os

 

# Tic-Tac-Toe Board Setup
def print_board(board):
    for row in [board[i:i + 3] for i in range(0, 9, 3)]:
        print("|".join(row))
        if row != board[-3:]:
            print("-----")

 

def check_winner(board, player):
    win_conditions = [(0, 1, 2), (3, 4, 5), (6, 7, 8), 
                      (0, 3, 6), (1, 4, 7), (2, 5, 8), 
                      (0, 4, 8), (2, 4, 6)]
    for condition in win_conditions:
        if all(board[i] == player for i in condition):
            return True
    return False

 

def is_draw(board):
    return all(spot != " " for spot in board)

 

# Q-learning AI functions
def get_available_moves(board):
    return [i for i, spot in enumerate(board) if spot == " "]

 

def get_best_action(state, q_table, available_moves):
    if state in q_table:
        # Get Q-values for all available moves
        q_values = {move: q_table[state].get(move, 0) for move in available_moves}
        print("AI is considering the following moves and their Q-values:", q_values)
        best_move = max(q_values, key=q_values.get)  # Choose the move with the highest Q-value
    else:
        best_move = random.choice(available_moves)  # If state is not in Q-table, pick randomly
    return best_move

 

def update_q_table(q_table, state, action, reward, next_state, alpha=0.1, gamma=0.9):
    if state not in q_table:
        q_table[state] = {}
    if action not in q_table[state]:
        q_table[state][action] = 0

 

    future_reward = max(q_table.get(next_state, {}).values(), default=0)
    q_table[state][action] += alpha * (reward + gamma * future_reward - q_table[state][action])

 

# Game and Training Logic
def play_game(q_table, is_training=True):
    board = [" " for _ in range(9)]
    player = "X"
    game_over = False
    states_actions = []  # To store the states and actions for Q-learning

 

    while not game_over:
        print_board(board)

 

        if player == "X":
            # Player move
            available_moves = get_available_moves(board)
            move = int(input(f"Choose your move {available_moves}: "))
        else:
            # AI move
            available_moves = get_available_moves(board)
            state = tuple(board)
            move = get_best_action(state, q_table, available_moves)

 

        board[move] = player
        states_actions.append((tuple(board), move))

 

        if check_winner(board, player):
            print_board(board)
            print(f"{player} wins!")
            if player == "X":
                reward = -1  # AI loses
            else:
                reward = 1  # AI wins
            game_over = True
        elif is_draw(board):
            print_board(board)
            print("It's a draw!")
            reward = 0
            game_over = True
        else:
            player = "O" if player == "X" else "X"
            continue

 

        # Update Q-table if training
        if is_training:
            for i, (state, action) in enumerate(states_actions):
                if i == len(states_actions) - 1:  # Last move
                    next_state = None
                else:
                    next_state = states_actions[i + 1][0]
                update_q_table(q_table, state, action, reward, next_state)

 

# Q-Table Management (Save and Load)
def save_q_table(q_table, filename="q_table.pkl"):
    with open(filename, 'wb') as file:
        pickle.dump(q_table, file)
    print("Q-table saved to", filename)

 

def load_q_table(filename="q_table.pkl"):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            q_table = pickle.load(file)
        print("Q-table loaded from", filename)
        return q_table
    else:
        print("No previous Q-table found, starting fresh.")
        return {}

 

# Main Function
def main():
    q_table = load_q_table()

 

    while True:
        print("\nTic-Tac-Toe Game!")
        choice = input("Do you want to train the AI (T) or play against it (P)? (T/P/Q to quit): ").upper()

 

        if choice == 'T':
            for _ in range(1000):  # Train for 1000 games
                play_game(q_table, is_training=True)
            print("Training complete.")
            save_q_table(q_table)

 

        elif choice == 'P':
            play_game(q_table, is_training=False)

 

        elif choice == 'Q':
            save_q_table(q_table)
            print("Exiting the game. Goodbye!")
            break

 

if __name__ == "__main__":
    main()

 

