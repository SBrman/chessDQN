import os
import time
import random
import pickle
import chess
import chess.svg

from typing import Union, final
from collections import deque, Counter
from tqdm import tqdm

from constants import *
from dqn_model import *
from moves import ALL_LEGAL_MOVES

import ipywidgets as wg
from IPython.display import display, SVG, clear_output

# Constants
TRAIN_INTERVAL = 2000


class ChessRL:
    def __init__(self, init_reward=None, q_method: QType = QType.DQN, experience_replay_size_threshold: int = 100000, 
                 saved_q: Union[str, os.PathLike, None] = 'dqn_py_chess'):
        
        # Initializing the board
        self.board = chess.Board()
        self.temp_board = chess.Board()
        
        # Initializing the rewards
        self._init_reward = init_reward
        self.reward_lb = -1
        self.reward_ub = 1
        
        # Initialize Q
        self.q_architecture = q_method
        self.dqn_model_path = saved_q
        self.initialize_q(saved_q)
        
        # Experience replay stuff
        self.experience_replay_visits = 0
        self.experience_replay = deque(maxlen=experience_replay_size_threshold)
        self.experience_replay_size_threshold = experience_replay_size_threshold
        
        # All possible actions (Including illegal ones)
        self.action_space_size = len(ALL_LEGAL_MOVES)
        
        # Q cache: Will change every time network is trained again
        self.q_cache = {}
        
        # Extra information to determine terminal state
        self._sub_episode = None
        self._max_episode_length = None
        
    def initialize_q(self, saved_q: Union[str, os.PathLike, None] = None):

        if self.q_architecture == QType.DQN:
            # Initializing the Deep Q network
            self.dqn = DeepQNetwork(saved_model=saved_q)

        elif self.q_architecture == QType.TABULAR:
            # Load weights from the already trained model
            self._Q = self.load_q_from_file(saved_q)
        else:
            raise NotImplementedError("Q type not implemented.")
    
    @property
    def init_reward(self):
        if not self._init_reward:
            self._init_reward = lambda: np.random.uniform(-1, 1)
        return self._init_reward()
    
    # Showing the board
    def show_board(self, board=None, size=BOARD_SIZE):
        if not board:
            board = self.board
            
        display(chess.svg.board(board, size=size))
        
    def game_real_time(self):
        """Displays the board and keeps the board positions"""
        while True:
            board, action = yield
            clear_output()
            display(chess.svg.board(board, size=BOARD_SIZE, lastmove=chess.Move.from_uci(action)))
    
    def display_next_board(self, action):
        gamert = self.game_real_time()
        next(gamert)
        gamert.send((self.board, action))
        
    def show_game(self, states, size=BOARD_SIZE):
        """Fix later"""
        t_board = chess.Board()
        boards = []
        for state in states:
            t_board.set_fen(state)
            boards.append(chess.svg.board(t_board, size=size))
        
        wg.interact(lambda x: display(boards[x]), x=wg.IntSlider(min=0, max=len(states)-1, step=1))
        
        del t_board
    
    # State space stuff
    def reset_to_initial_state(self):
        self.board.reset()
    
    @property
    def current_state(self):
        return str(self.board.fen())

    @property
    def turn(self):
        return "white" if self.board.turn else "black"
    
    @property 
    def terminal(self):
        return self.board.is_game_over()
    
    def simplify_state(self, state: str) -> str:
        """Returns only the position of pieces, everything else is removed from fen."""
        self.temp_board.set_fen(state)
        return f"{self.temp_board}".replace(' ', '').replace('\n', '')
    
    def num_rep_state(self, state: str) -> np.array:
        """
        Returns the numerical representation of the board state. 
        Sum can be used for static evaluation of the board.
        This will be used as the input to the neural net.
        """
        state = self.simplify_state(state)
        return np.array([PIECE_MAP[piece] for piece in state]).reshape(8, 8)
   
    # Action stuff 
    def num_rep_action_q_values(self, state: str, action: str = None, value: float = None):
        """
        Returns the Q values for the actions at a given state. 
        This will be used as the label for a given state to train the neural net.
        """
        one_hot_action_Qs = np.zeros(len(ALL_LEGAL_MOVES))
        
        action_qs = {ALL_LEGAL_MOVES[action]: self.normalized_Q(state, action) 
                     for action in self.possible_actions(state)}
        action_qs[ALL_LEGAL_MOVES[action]] = value

        np.put(one_hot_action_Qs, list(action_qs), list(action_qs.values()))
        return one_hot_action_Qs
    
    def possible_actions(self, state):
        self.temp_board.set_fen(state)
        return [str(move) for move in self.temp_board.legal_moves]
    
    def choose_greedy_action(self, epsilon: float) -> bool:
        return True if np.random.uniform(0, 1) < (1 - epsilon) else False
        
    def epsilon_greedy_action(self, state, epsilon):
        if self.choose_greedy_action(epsilon):
            return self.argmax_Q(state)
        else:
            actions = self.possible_actions(state)
            random_action_index = random.randint(0, len(actions)-1)
            return actions[random_action_index]
            
    def take_action(self, action: str):
        # Taking action
        self.board.push_uci(action)
        
        # Get the new state because of taken action
        new_state = self.current_state
        
        # Get the reward because of the taken action
        reward = self.reward
        
        return (new_state, reward)
    
    # Reward Stuff
    @property
    def reward(self):
        if self.terminal:
            if self.board.result in ["1-0", "0-1"]:
                return 1
            else:
                return -1
        return 0
        
    
    # Q stuff
    def update_q_cache(self, state, simple_state):
        numerical_state = self.num_rep_state(state)
        normalized_q = self.dqn.get_Q(numerical_state)
        self.q_cache[simple_state] = self.denormalize_Q(normalized_q)
        
    def Q(self, state, action):
        """
        Return the Q values from the Deep Q Network if DQN is used otherwise return q values from the lookup table.
        """
        action = ALL_LEGAL_MOVES[action]
        simple_state = self.simplify_state(state) if isinstance(state, dict) else state
        if simple_state in self.q_cache:
            return self.q_cache[simple_state][action]
        
        if self.q_architecture == QType.DQN:
            # DQN will predict the q values
            self.update_q_cache(state, simple_state)
            return self.q_cache[simple_state][action]

        else: 
            # If q architecture is not a nn then q values will be obtained from the q lookup table.
            if simple_state not in self._Q:
                for a in self.possible_actions(state):
                    self.set_Q(state=state, action=a, value=self.init_reward)
                    
            elif action not in self._Q[simple_state]:
                self._Q[simple_state][action] = self.init_reward
                
            return self._Q[simple_state][action]
    
    def normalized_Q(self, state, action):
        return (self.Q(state, action) - self.reward_lb) / (self.reward_ub - self.reward_lb)
    
    def denormalize_Q(self, normalized_q_value):
        return normalized_q_value * (self.reward_ub - self.reward_lb) + self.reward_lb

    def set_Q(self, state, action, value=None):
        
        if self.q_architecture == QType.DQN:
            # Keep accumulating data
            state_feature = self.num_rep_state(state)
            action_label = self.num_rep_action_q_values(state, action, value)
            self.experience_replay.append((state_feature, action_label))
            self.experience_replay_visits += 1

            # Train dqn with this accumulated data if have enough data
            train_interval_ok = not self.experience_replay_visits % TRAIN_INTERVAL
            have_enough_training_data = len(self.experience_replay) == self.experience_replay_size_threshold

            if train_interval_ok and have_enough_training_data:            
                # Train if enough data accumulated
                self.dqn.train(self.experience_replay, save_path=self.dqn_model_path)
                self.q_cache = {}

        else:
            state = self.simplify_state(state)
            self._Q.setdefault(state, {})
            self._Q[state][action] = value
        
    def max_Qa(self, state):
        return max([(self.Q(state, action), action) for action in self.possible_actions(state)])
        
    def max_Q(self, state):
        """Returns the max Q value for the best action."""
        return self.max_Qa(state)[0]
    
    def argmax_Q(self, state):
        """Returns the best action for which max Q was found for the input state."""
        return self.max_Qa(state)[1]
        

    # Learning Stuff
    def q_learning(self, episodes, epsilon, alpha, gamma, max_episode_length, show_interval):
        # Initialize progress bar
        # f = wg.IntProgress(min=0, max=episodes)
        # display(f)
        
        for episode in tqdm(range(episodes), desc='Episodes'):
            try:
                # Show progress
                print(f'{episode = }')
                # f.value = episode

                # Initialize a state
                self.reset_to_initial_state()
                state = self.current_state
                
                # Storing just to visualize games
                # play_states = [state]
                episode_length = 0

                progress_bar = tqdm(total=max_episode_length)
                
                while not self.terminal and episode_length <= max_episode_length:
    ##                print(f"{len(play_states)}", end=', ')
                    progress_bar.update(1)

                    # Select epsilon greedy action
                    action = self.epsilon_greedy_action(state, epsilon)
                    
                    # Take action get reward and new state
                    new_state, reward = self.take_action(action)
                    
                    # Storing just to visualize games
                    episode_length += 1 
                    # play_states.append(new_state)
                    
                    if not self.terminal:
                        # Q-learning update
                        off_policy_td = (reward + gamma * self.max_Q(new_state) - self.Q(state, action))
                        new_Q_sa = self.Q(state, action) + alpha * off_policy_td
                    else:
                        new_Q_sa = reward
                    
                    # Update the DQN model or the Q lookup table
                    self.set_Q(state, action, new_Q_sa)
                    
                    # Update the state
                    state = new_state        
                
                # if episode % show_interval == 0:
                #     print(len(play_states))
                #     self.show_game(states=play_states)
                #     del play_states


            # TODO: figure out whats wrong later?
            except Exception as e:
                send_msg(str(e))
                self.experience_replay.pop()
                continue

        self.save_q_to_file('dqn')
        
        
    def sarsa(self, episodes, epsilon, alpha, gamma, max_episode_length, show_interval):    
        # Initialize progress bar
        # f = wg.IntProgress(min=0, max=episodes)
        # display(f)
        
        for episode in range(episodes):
            
            # Show progress
            # f.value = episode

            # Initialize a state
            self.reset_to_initial_state()
            state = self.current_state
            
            # Storing just to visualize a game
            play_states = [state]

            # Select epsilon greedy action
            action = self.epsilon_greedy_action(state, epsilon)
            
            while not self.terminal and len(play_states) <= max_episode_length:
                
                # Take action get reward and new state
                new_state, reward = self.take_action(action)
                
                # Storing just to visualize games
                play_states.append(new_state)
                
                if not self.terminal:
                    # SARSA update
                    new_action = self.epsilon_greedy_action(new_state, epsilon)
                    on_policy_td = (reward + gamma * self.Q(new_state, new_action) - self.Q(state, action))
                    new_Q_sa = self.Q(state, action) + alpha * on_policy_td
                else:
                    new_Q_sa = reward
                
                # Update the DQN model or the Q lookup table
                self.set_Q(state, action, new_Q_sa)
                
                # Update state and action
                state = new_state
                action = new_action
            
            if episode % show_interval == 0:
                print(len(play_states))
                self.show_game(states=play_states)
                del play_states
        
        self.save_q_to_file('sarsa_nn')

    
    def truncated_rollout(self, episodes, epsilon, alpha, gamma, truncation_depth, max_episode_length, show_interval):
        pass
    
    def monte_carlo_tree_search(self, episodes, epsilon, alpha, gamma, truncation_depth, max_episode_length, show_interval):    
        pass
           
           
    # Heuristic policy
    def minimax_with_alpha_beta_pruning(self):
        pass
    
    # Static Evaluation of game state
    def remaining_materials(self, state):
        return Counter(str(self.simplify_state(state)))
    
    def material_score(self, state):
        weighted_materials = self.num_rep_state(state)

        # Replace the knights scores, which is set to 4, to 3
        weighted_materials[weighted_materials == 4] = 3

        # Replace the kings scores from 10 to 200 according to Claude Shannon's formulation 
        weighted_materials[weighted_materials == 10] = 200

        return np.sum(weighted_materials)
    
    def mobility_score(self, state):
        """
        Returns the mobility score at state s 
        mobility score(s) = current players legal moves(s) - opponents legal moves(s)
        """
        # Create a temporary board to check the legal moves number at different states 
        # without affecting the actual game.
        self.temp_board.set_fen(state)
        
        # Get the current player's number legal moves
        current_player_moves = len(self.temp_board.legal_moves)

        # Pass control to the opponent
        self.temp_board.push(chess.Move.null())

        # Get the opponent player's number of legal move 
        opponents_moves = len(self.temp_board.legal_moves)
        
        return current_player_moves - opponents_moves
        
    def static_analysis_of_board(self, state):
        """Returns the positional evaluation of the board."""
        return self.material_score(state) + self.mobility_score(state) 
    
    # Saving and loading q values 
    def save_q_to_file(self, filename: str):
        filename += '_py_chess'
        if self.q_architecture == QType.DQN:
            self.dqn.model.save(filename)

        elif self.q_architecture == QType.TABULAR: 
            with open(filename, 'wb') as file:
                pickle.dump(self._Q, file, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            raise NotImplementedError("Unknown Q architecture")
        
    def load_q_from_file(self, filename: str):

        if self.q_architecture == QType.DQN:
            dqn_model = None
            try:
                dqn_model = keras.models.load_model(filename)
            except FileNotFoundError:
                print('File does not exist')
            except OSError:
                print('File does not exist')

            return dqn_model

        elif self.q_architecture == QType.TABULAR:
            q_lookup_table = {}
            try:
                with open(filename, 'rb') as file:
                    q_lookup_table = pickle.load(file)
            except FileNotFoundError:
                pass
            return q_lookup_table

        else:
            raise NotImplementedError("Unknown Q architecture")
    
    def play(self):
        self.reset_to_initial_state()
        state = self.current_state
        states = [state]
        while not self.terminal:
            action = self.argmax_Q(state)
            state, _ = self.take_action(action)
            states.append(state)
        self.show_game(states=states)
