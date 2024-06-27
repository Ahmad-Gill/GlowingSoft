import numpy as np
import tensorflow as tf
import random
from tensorflow import keras
import chess
import chess.pgn
import io
import os
import datetime

PGN_PATH = './lichess_db_standard_rated_2015-08.pgn'
# MODEL_PATH = './chess_model.h5'

class ChessAI:
    def __init__(self, MODEL_PATH=""):
        if os.path.exists(MODEL_PATH):
            self.model = keras.models.load_model(MODEL_PATH)
            print("Model loaded from disk.")
            # Recreate the optimizer
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            self.model = self.create_model()

        self.log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        
        # Initialize reinforcement learning parameters
        self.memory = []
        self.gamma = 0.95  # discount factor
        self.epsilon = 0.1  # exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    def board_to_input(self, board):
        piece_chars = 'PRNBQKprnbqk'
        input_vector = np.zeros(64 * 12)
        for i in range(64):
            piece = board.piece_at(i)
            if piece:
                input_vector[i * 12 + piece_chars.index(piece.symbol())] = 1
        return input_vector

    def get_best_move(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        input_vector = self.board_to_input(board)
        
        if np.random.rand() <= self.epsilon:
            return np.random.choice(legal_moves)
        
        predictions = self.model.predict(np.array([input_vector]))[0]
        
        move_scores = [(move, predictions[move.from_square]) for move in legal_moves]
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        best_move = move_scores[0][0] if move_scores else None
        return best_move
    def create_model(self):
        model = keras.Sequential([
            keras.layers.Dense(256, activation='relu', input_shape=(64 * 12,)),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model


    def train_on_pgn(self, pgn_file, start_game=0, end_game=0):
        print("Starting training from game:", start_game, "to game:", end_game)
        is_trained = False
        game_number = 0

        with open(pgn_file) as f:
            print(f"TRAINING FILE {pgn_file} HAS BEEN OPENED")

            # Skip games until reaching the start_game
            while game_number < start_game:
                game_ = chess.pgn.read_game(f)
                if game_ is None:
                    break
                game_number += 1

            # Process games until reaching end_game
            while game_number < end_game:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                board = game.board()
                for move in game.mainline_moves():
                    input_vector = self.board_to_input(board)
                    target = np.zeros(64)
                    target[move.from_square] = 1
                    self.model.fit(np.array([input_vector]), np.array([target]), verbose=0, callbacks=[self.tensorboard_callback])

                    move_made = move.__str__()
                    board.push(move)

                    yield (game_number + 1), is_trained, move_made

                game_number += 1

        is_trained = True
        yield (game_number + 1), is_trained, None




# def train_on_pgn(self, pgn_file, start_game=0, num_games=1000):
#     is_trained = False
#     with open(pgn_file) as f:
#         print(f"TRAINING FILE {pgn_file} HAS BEEN OPENED")
#         game_number = 0
        
#         while game_number < start_game:
#             game = chess.pgn.read_game(f)
#             if game is None:
#                 break
#             game_number += 1
        
#         for game_number in range(start_game, start_game + num_games):
#             game = chess.pgn.read_game(f)
#             if game is None:
#                 break
#             board = game.board()
#             for move in game.mainline_moves():
#                 input_vector = self.board_to_input(board)
#                 target = np.zeros(64)
#                 target[move.from_square] = 1
#                 self.model.fit(np.array([input_vector]), np.array([target]), verbose=0, callbacks=[self.tensorboard_callback])
                
#                 move_made = move.__str__()
#                 board.push(move)
                
#                 yield (game_number+1), is_trained, move_made
#     is_trained = True
#     yield (game_number+1), is_trained, move

        
        # self.save_model()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_input_vector = self.board_to_input(next_state)
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_input_vector]))[0])
            
            input_vector = self.board_to_input(state)
            target_f = self.model.predict(np.array([input_vector]))
            target_f[0][action.from_square] = target
            
            self.model.fit(np.array([input_vector]), target_f, epochs=1, verbose=0)
            print("Moved Learned")

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, MODEL_PATH):
        self.model.save(MODEL_PATH)
        print("Model saved to disk.")
    # def save_model(self, MODEL_PATH):
    #     # Ensure the file has a .keras extension
    #     if not MODEL_PATH.endswith('.keras'):
    #         MODEL_PATH = MODEL_PATH.rsplit('.', 1)[0] + '.keras'
        
    #     self.model.save(MODEL_PATH, save_format='keras')
    #     print(f"Model saved to disk at {MODEL_PATH}")

    def remember(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        input_vector = self.board_to_input(board)
        
        if np.random.rand() <= self.epsilon:
            return np.random.choice(legal_moves)
        
        predictions = self.model.predict(np.array([input_vector]))[0]
        
        move_scores = [(move, predictions[move.from_square]) for move in legal_moves]
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        best_move = move_scores[0][0] if move_scores else None
        return best_move


# Function to convert ChessState to chess.Board
def chess_state_to_board(gs):
    fen = ""
    for row in gs.board:
        empty = 0
        for piece in row:
            if piece == "--":
                empty += 1
            else:
                if empty > 0:
                    fen += str(empty)
                    empty = 0
                fen += piece[1].lower() if piece[0] == 'b' else piece[1].upper()
        if empty > 0:
            fen += str(empty)
        fen += "/"
    fen = fen[:-1]  # remove last slash
    fen += " w KQkq - 0 1"  # Add default values for now
    return chess.Board(fen)


# In your main game loop, when it's AI's turn:
# board = chess_state_to_board(gs)  # Convert your GameState to chess.Board
# best_move = ai.get_best_move(board)
# if best_move:
#     # Convert best_move to your Move class and make the move
#     # For example: gs.makeMove(Move(best_move.from_square, best_move.to_square, gs.board))

if __name__ == "__main__":
    # Usage example
    ai = ChessAI()
    ai.train_on_pgn(PGN_PATH)  # Train the AI on PGN data

