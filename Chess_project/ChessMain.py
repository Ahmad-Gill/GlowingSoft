import pygame as p
from sympy import python
import yaml
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
from Chess import ChessState
from ChessAI import ChessAI, chess_state_to_board
import queue 
import threading
import subprocess
import sys

WIDTH = HEIGHT = 512
DIMENSION = 8  # 8*8 board
SQ_SIZE = HEIGHT // DIMENSION
MAX_FPS = 15
IMAGES = {}
BOARDER_SIZE = 40
LABEL_FONT_SIZE = 20
NUM_GAMES_TRAIN = 5
PGN_PATH = './lichess_db_standard_rated_2015-08.pgn'
CONFIG_PATH = './config.yaml'

# Load configuration
def load_config():
    
    default_config = {'model_path': './chess_model.h5', 'use_checkpoint': True}

    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file) or default_config
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file '{CONFIG_PATH}': {e}")
            return default_config
        except Exception as e:
            print(f"Error loading config file '{CONFIG_PATH}': {e}")
            return default_config
    else:
        save_config(default_config)
        return default_config

# Save configuration
def save_config(config):
    print("the data is save to file")
    with open(CONFIG_PATH, 'w', encoding='utf-8') as file:
        yaml.safe_dump(config, file)

config = load_config()
model_path = config['model_path']

# Initialize global dictionary of images
def loadImages():
    pieces = ['wp', 'wR', 'wN', 'wB', 'wK', 'wQ', 'bp', 'bR', 'bN', 'bB', 'bK', 'bQ']
    for piece in pieces:
        IMAGES[piece] = p.transform.scale(p.image.load("images/" + piece + ".png"), (SQ_SIZE, SQ_SIZE))

def train_ai(ai):
    try:
        game_number_last = 1
        move_count = 0
        with open('txt.txt', 'r') as file:
            start = int(file.readline().strip())
        moves_train = ai.train_on_pgn(PGN_PATH, start_game=start, end_game=NUM_GAMES_TRAIN)

        for game_number, is_trained, move in moves_train:
            
            move_count += 1
            if game_number == game_number_last + 1:
                move_count = 0
                game_number_last = game_number
            
            visualize = move_count <= 1000
            print("Game number: ", game_number, "move_number: ", move_count)
            
            yield visualize, is_trained, move
            if is_trained:
                ai.save_model(config['model_path'])
                save_config(config)
                break
    except FileNotFoundError:
        print("ERROR: Training file not found or inaccessible")
        yield None

def train_():
    with open('txt.txt', 'r') as file:
        single_value = int(file.readline().strip())
        if NUM_GAMES_TRAIN <=single_value:
            return False
        else:
            return True
        



def main():
    p.init()
    screen = p.display.set_mode((WIDTH, HEIGHT))
    clock = p.time.Clock()
    screen.fill(p.Color("white"))
    gs = ChessState.GameState()
    validMoves = gs.getValidMoves()
    moveMade = False  # flag var for when a move is made
    animate = False   # flag variable for when we should animate a move
    loadImages()
    running = True
    is_training = True
    training_error = False
    sqSelected = ()
    playerClicks = []
    gameOver = False
    playerOne = True  # If a human is playing white, else False
    playerTwo = False  # If a human is playing black, else False

    if os.path.exists(config['model_path']) and not config['use_checkpoint']:
        ai = ChessAI(MODEL_PATH=config['model_path'])
        is_training = train_()
        if is_training:
            print("Loading the existing model for training on new data")
            train_generator = train_ai(ai)

    elif os.path.exists(config['model_path']) and config['use_checkpoint']:
        ai = ChessAI(MODEL_PATH=config['model_path'])
        is_training = train_()
        if is_training:
            print("Loading the existing model for training on new data")
            train_generator = train_ai(ai)

    else:
        print("Starting the training from scratch")
        ai = ChessAI()
        is_training = train_()
        if is_training:
            print("Loading the existing model for training on new data")
            train_generator = train_ai(ai) # Call train_ai directly instead of using multiprocessing

    batch_size = 32
    game_memory = []

    while running:
        if is_training:
            try:
                move_none = ChessState.Move((0, 0), (0, 0), gs.board)
                move_data = next(train_generator)
                if move_data is None:
                    training_error = True
                    is_training = False
                    continue
                visualize, is_trained, move = move_data
                if not is_trained:
                    if visualize:
                        move_player = move_none.fromChessNotation(move, gs.board)
                        for i in range(len(validMoves)):
                            if move_player == validMoves[i]:
                                gs.makeMove(validMoves[i])
                                moveMade = True
                                animate = True
                else:
                    is_training = False
            except queue.Empty: 
                pass
        if train_()  and not is_training:
            save_config(config)
            with open('txt.txt', 'w') as file:
                file.write(str(NUM_GAMES_TRAIN))
            print('the model is saved successfully to use the complete setup with more knowledge of moves restart and it it load from disk with new values')
            python = sys.executable
            subprocess.Popen([python] + sys.argv)
            sys.exit()
        if not gameOver and not is_training and not gs.checkMate and not gs.staleMate:
            humanTurn = (gs.whiteToMove and playerOne) or (not gs.whiteToMove and playerTwo)
            if humanTurn:
                for e in p.event.get():
                    if e.type == p.QUIT:
                        running = False
                    elif e.type == p.KEYDOWN:
                        if e.key == p.K_ESCAPE:
                            is_training = False
                        elif e.key == p.K_z:  # undo when z is pressed
                            gs.undoMove()
                            gs.undoMove()
                            moveMade = True
                            animate = False
                        elif e.key == p.K_r:  # reset game when 'r' is pressed
                            gs = ChessState.GameState()
                            validMoves = gs.getValidMoves()
                            sqSelected = ()
                            playerClicks = []
                            moveMade = False
                            animate = False
                            gameOver = False
                    elif e.type == p.MOUSEBUTTONDOWN:
                        if not gameOver and humanTurn:
                            location = p.mouse.get_pos()  # (x, y) location of mouse
                            col = location[0] // SQ_SIZE
                            row = location[1] // SQ_SIZE
                            if sqSelected == (row, col):  # user click same square twice
                                sqSelected = ()  # deselect
                                playerClicks = []
                            else:
                                sqSelected = (row, col)
                                playerClicks.append(sqSelected)  # for both first and second clicks
                            if len(playerClicks) == 2:  # after second click
                                move = ChessState.Move(playerClicks[0], playerClicks[1], gs.board)
                                print(f"Human move attempted: {move.getChessNotation()}")
                                for i in range(len(validMoves)):
                                    if move == validMoves[i]:
                                        gs.makeMove(validMoves[i])
                                        moveMade = True
                                        animate = True
                                        sqSelected = ()  # reset user click
                                        playerClicks = []
                                if not moveMade:
                                    playerClicks = [sqSelected]
            else:  # AI's turn
                board = chess_state_to_board(gs)
                current_state = board
                
                ai_move = ai.get_best_move(board)
                
                if ai_move:
                    ai_move_made = False
                    for move in validMoves:
                        if move.startRow == ai_move.from_square // 8 and move.startCol == ai_move.from_square % 8 and \
                                move.endRow == ai_move.to_square // 8 and move.endCol == ai_move.to_square % 8:
                            gs.makeMove(move)
                            moveMade = True
                            animate = True
                            ai_move_made = True
                            break
                    
                    if ai_move_made:
                        next_board = chess_state_to_board(gs)
                        reward = 0.1  # Small positive reward for making a move
                        
                        if gs.checkMate:
                            reward = 1 if gs.whiteToMove else -1
                        elif gs.staleMate:
                            reward = 0
                        
                        game_memory.append((current_state, ai_move, reward, next_board, gs.checkMate or gs.staleMate))
                        
                        if len(game_memory) >= batch_size:
                            for state, action, reward, next_state, done in game_memory:
                                ai.remember(state, action, reward, next_state, done)
                            ai.replay(batch_size)
                            game_memory = []
                
                if not ai_move_made:
                    import random
                    if validMoves:
                        random_move = random.choice(validMoves)
                        gs.makeMove(random_move)
                        moveMade = True
                        animate = True
                    else:
                        print("No valid moves available. Game over.")
                        gameOver = True

        if moveMade:
            if animate:
                animateMove(gs.moveLog[-1], screen, gs.board, clock)
            validMoves = gs.getValidMoves()
            moveMade = False
            animate = False

        drawGameState(screen, gs, validMoves, sqSelected)

        if gs.checkMate and not is_training:
            gameOver = True
            if gs.whiteToMove:
                drawText(screen, 'Black wins by checkmate')
            else:
                drawText(screen, 'White wins by checkmate')
        elif gs.staleMate:
            gameOver = True
            drawText(screen, 'Stalemate')

        if is_training:
            drawText(screen, 'Training...')
        # elif training_error:
        #     drawText(screen, 'Training enabled but data not found, playing human vs human')
        #     running = True

        # Save model periodically
        if not is_training and not gameOver and gs.moveCount % 100 == 0:
            ai.save_model(config['model_path'])

        clock.tick(MAX_FPS)
        p.display.flip()

    # Save model at the end of the game
    if not is_training:
        ai.save_model(config['model_path'])

def highlightSquares(screen, gs, validMoves, sqSelected):
    if sqSelected != ():
        r, c = sqSelected
        if gs.board[r][c][0] == ('w' if gs.whiteToMove else 'b'):  # sqSelected is a piece that can be move
            s = p.Surface((SQ_SIZE, SQ_SIZE))
            s.set_alpha(100)  # transparent value
            s.fill(p.Color('blue'))
            screen.blit(s, (c * SQ_SIZE, r * SQ_SIZE))
            s.fill(p.Color('yellow'))
            for move in validMoves:
                if move.startRow == r and move.startCol == c:
                    screen.blit(s, (move.endCol * SQ_SIZE, move.endRow * SQ_SIZE))

def drawGameState(screen, gs, validMoves, sqSelected):
    drawBoard(screen)  # draw square on the board
    highlightSquares(screen, gs, validMoves, sqSelected)
    drawPieces(screen, gs.board)  # draw pieces on top of those squares

def drawBoard(screen):
    global colors
    colors = [p.Color("white"), p.Color("gray")]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[((r + c) % 2)]
            p.draw.rect(screen, color, p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def drawPieces(screen, board):
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            piece = board[r][c]
            if piece != "--":
                screen.blit(IMAGES[piece], p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def animateMove(move, screen, board, clock):
    global colors
    dR = move.endRow - move.startRow
    dC = move.endCol - move.startCol
    framesPerSquare = 10  # frames to move one square
    frameCount = (abs(dR) + abs(dC)) * framesPerSquare
    for frame in range(frameCount + 1):
        r, c = (move.startRow + dR * frame / frameCount, move.startCol + dC * frame / frameCount)
        drawBoard(screen)
        drawPieces(screen, board)
        # erase the piece moved from its ending square
        color = colors[(move.endRow + move.endCol) % 2]
        endSquare = p.Rect(move.endCol * SQ_SIZE, move.endRow * SQ_SIZE, SQ_SIZE, SQ_SIZE)
        p.draw.rect(screen, color, endSquare)
        # draw captured piece onto rectangle
        if move.pieceCaptured != '--':
            screen.blit(IMAGES[move.pieceCaptured], endSquare)
        # draw moving piece
        screen.blit(IMAGES[move.pieceMoved], p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))
        p.display.flip()
        clock.tick(60)

def drawText(screen, text):
    font = p.font.SysFont("Helvitca", 32, True, False)
    textObject = font.render(text, 0, p.Color('Gray'))
    textLocation = p.Rect(0, 0, WIDTH, HEIGHT).move(WIDTH / 2 - textObject.get_width() / 2,
                                                    HEIGHT / 2 - textObject.get_height() / 2)
    screen.blit(textObject, textLocation)
    textObject = font.render(text, 0, p.Color("Black"))
    screen.blit(textObject, textLocation.move(2, 2))

if __name__ == "__main__":
    main()