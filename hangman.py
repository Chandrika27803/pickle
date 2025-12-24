import tkinter as tk
import random
import requests
 
# ---------------------------------------------------------
# Get a random English word from online API
# ---------------------------------------------------------
def get_random_word():
    """
    Fetch a random word from an online API.
    Falls back to 'PYTHON' if there is any error.
    """
    try:
        response = requests.get("https://random-word-api.herokuapp.com/word?number=1", timeout=3)
        if response.status_code == 200:
            word = response.json()[0]
            return word.upper()
    except Exception:
        pass
 
    return "PYTHON"  # fallback word
 
 
# Number of wrong guesses allowed
MAX_ATTEMPTS = 6

def get_clue(word):
    try:
        url=f"https://api.dictionaryapi.dev/api/v2/entries/en/{word.lower()}"
        response=requests.get(url,timeout=3)
        if response.status_code==200:
            data=response.json()
            if isinstance(data,list) and data:
                meanings=data[0].get("meanings",[])
                if meanings:
                    definitions=meanings[0].get("definitions",[])
                    if definitions:
                        definition=definitions[0].get("definition","")
                        if definition:
                            if len(definition)>120:
                                definition=definition[:117]+"..."
                                return definition
    except Exception:
        pass
        return f"No clue found for the {word}"
        
 
class HangmanGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Hangman - Tkinter Edition (Random Online Words)")
 
        # Game state variables
        self.secret_word = ""
        self.guessed_letters = set()
        self.wrong_letters = set()
        self.remaining_attempts = MAX_ATTEMPTS
 
        # ---------- UI ----------
        top_frame = tk.Frame(root)
        top_frame.pack(pady=10)
 
        self.word_label = tk.Label(top_frame, text="Press 'New Game' to begin", font=("Arial", 20))
        self.word_label.pack()
        self.clue_label=tk.Label(top_frame, text="Clue", font=("Arial", 20))
 
        self.status_label = tk.Label(root, text="", font=("Arial", 12))
        self.status_label.pack(pady=5)
 
        middle_frame = tk.Frame(root)
        middle_frame.pack(pady=10)
 
        # Canvas for drawing hangman
        self.canvas = tk.Canvas(middle_frame, width=200, height=250, bg="white")
        self.canvas.grid(row=0, column=0, padx=10)
 
        # Info panel
        info_frame = tk.Frame(middle_frame)
        info_frame.grid(row=0, column=1, padx=10)
 
        self.wrong_label = tk.Label(info_frame, text="Wrong letters: ", font=("Arial", 12))
        self.wrong_label.pack(anchor="w", pady=5)
 
        self.attempts_label = tk.Label(info_frame, text=f"Remaining attempts: {MAX_ATTEMPTS}", font=("Arial", 12))
        self.attempts_label.pack(anchor="w", pady=5)
 
        # Bottom controls
        bottom_frame = tk.Frame(root)
        bottom_frame.pack(pady=10)
 
        tk.Label(bottom_frame, text="Enter a letter:", font=("Arial", 12)).grid(row=0, column=0, padx=5)
 
        self.entry = tk.Entry(bottom_frame, width=5, font=("Arial", 14))
        self.entry.grid(row=0, column=1, padx=5)
        self.entry.bind("<Return>", self.handle_guess_event)
 
        self.guess_button = tk.Button(bottom_frame, text="Guess", font=("Arial", 12),
                                      command=self.handle_guess)
        self.guess_button.grid(row=0, column=2, padx=5)
 
        # New and reset game buttons
        self.new_game_button = tk.Button(root, text="New Game", font=("Arial", 12),
                                         command=self.start_new_game)
        self.new_game_button.pack(pady=10)
 
        self.reset_button = tk.Button(root, text="Reset Game", font=("Arial", 12),
                                      command=self.reset_board)
        self.reset_button.pack()
 
        self.set_guess_enabled(False)
 
    # ---------- GAME LOGIC ----------
 
    def start_new_game(self):
        """Start a new game."""
        self.secret_word = get_random_word()
        print("New random word:", self.secret_word)  # helpful debug
        clue=get_clue(self.secret_word)
        #print(clue)
        self.guessed_letters = set()
        self.wrong_letters = set()
        self.remaining_attempts = MAX_ATTEMPTS
 
        self.update_word_display()
        self.status_label.config(text="Game started! Guess a letter.")
        self.wrong_label.config(text="Wrong letters: ")
        self.attempts_label.config(text=f"Remaining attempts: {self.remaining_attempts}")
        self.clue_label.config(text=f"Clue: {clue}")
        print(clue)
 
        self.clear_canvas()
        self.draw_gallows()
 
        self.set_guess_enabled(True)
        self.entry.focus_set()
 
    def update_word_display(self):
        """Update on-screen underscore display."""
        display = [
            ch if ch in self.guessed_letters else "_"
            for ch in self.secret_word
        ]
        self.word_label.config(text=" ".join(display))
 
    def handle_guess_event(self, event):
        self.handle_guess()
 
    def handle_guess(self):
        """Process a single-letter guess."""
        if self.remaining_attempts <= 0:
            return
 
        guess = self.entry.get().strip().upper()
        self.entry.delete(0, tk.END)
 
        if len(guess) != 1 or not guess.isalpha():
            self.status_label.config(text="Enter a single letter.")
            return
 
        if guess in self.guessed_letters or guess in self.wrong_letters:
            self.status_label.config(text=f"You already guessed '{guess}'.")
            return
 
        if guess in self.secret_word:
            self.guessed_letters.add(guess)
            self.status_label.config(text=f"Good! '{guess}' is correct.")
            self.update_word_display()
            self.check_win()
        else:
            self.wrong_letters.add(guess)
            self.remaining_attempts -= 1
            self.status_label.config(text=f"Wrong! '{guess}' is not in the word.")
            self.wrong_label.config(text=f"Wrong letters: {', '.join(sorted(self.wrong_letters))}")
            self.attempts_label.config(text=f"Remaining attempts: {self.remaining_attempts}")
            self.draw_hangman_stage()
            self.check_loss()
 
    def check_win(self):
        """Check if the player guessed all letters."""
        if all(ch in self.guessed_letters for ch in self.secret_word):
            self.status_label.config(text=f"🎉 YOU WIN! Word: {self.secret_word}")
            self.set_guess_enabled(False)
 
    def check_loss(self):
        """Check if player has lost."""
        if self.remaining_attempts <= 0:
            self.status_label.config(text=f"💀 YOU LOSE! Word was: {self.secret_word}")
            self.word_label.config(text=" ".join(self.secret_word))  # reveal word
            self.set_guess_enabled(False)
 
    def set_guess_enabled(self, enabled):
        """Enable/disable guess input."""
        state = tk.NORMAL if enabled else tk.DISABLED
        self.entry.config(state=state)
        self.guess_button.config(state=state)
 
    # ---------- DRAWING ----------
 
    def clear_canvas(self):
        self.canvas.delete("all")
 
    def draw_gallows(self):
        """Draw the static gallows."""
        self.canvas.create_line(20, 230, 180, 230, width=3)
        self.canvas.create_line(50, 230, 50, 20, width=3)
        self.canvas.create_line(50, 20, 130, 20, width=3)
        self.canvas.create_line(130, 20, 130, 50, width=3)
 
    def draw_hangman_stage(self):
        wrong = len(self.wrong_letters)
 
        if wrong >= 1:
            self.canvas.create_oval(110, 50, 150, 90, width=2)
        if wrong >= 2:
            self.canvas.create_line(130, 90, 130, 150, width=2)
        if wrong >= 3:
            self.canvas.create_line(130, 110, 110, 130, width=2)
        if wrong >= 4:
            self.canvas.create_line(130, 110, 150, 130, width=2)
        if wrong >= 5:
            self.canvas.create_line(130, 150, 115, 190, width=2)
        if wrong >= 6:
            self.canvas.create_line(130, 150, 145, 190, width=2)
 
    def reset_board(self):
        """Clear the board only."""
        self.clear_canvas()
        self.draw_gallows()
        self.entry.delete(0, tk.END)
 
 
# ---------- RUN GAME ----------
if __name__ == "__main__":
    root = tk.Tk()
    HangmanGame(root)
    root.mainloop()
