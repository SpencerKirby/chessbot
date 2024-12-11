"""
Spencer Kirby
machine learning neural network that learns of chess Very slow not optimized well 
absolutly horrible at chess 
4/25/24 
"""
import gc#imports stuff
import numpy as np 
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from datetime import datetime as dt

class ChessEnv:#chess envirnment
    def __init__(self):  # creates the object with starting board 
        self.reset()

    def flip(self):  # flips the board 
        for i in range(64):
            self.state[i] *= -1

    def reset(self):  # resets chess board to starting position
        self.state = [
            -2, -3, -4, -5, -6, -4, -3, -2,  
            -1, -1, -1, -1, -1, -1, -1, -1,  
            0, 0, 0, 0, 0, 0, 0, 0,  
            0, 0, 0, 0, 0, 0, 0, 0,  
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,  
            1, 1, 1, 1, 1, 1, 1, 1,  
            2, 3, 4, 5, 6, 4, 3, 2,  
            False, False, False, False, False, False  # Castling flags
        ]
        self.done = False
        self.winner = None

    def is_valid_square(self, index):# checks if index is a square 
        
        return 0 <= index < 64

    def move(self, start, end):#moves the peice if legal move /updates flags 
        
        if not self.is_valid_square(start) or not self.is_valid_square(end) or not self.is_valid_move(start, end) :
            return -100  # Large penalty for an illegal move
        piece = self.state[start]
        
        if  piece < 0:
            return -100 # large penelty for moving black peices

        # Perform the move
        self.state[start] = 0
        self.state[end] = piece

        # Check for game-ending conditions and gives points 
        if self.over() == 1:  
            self.winner = 1
            return 1000  
        elif self.over() == -1: 
            self.winner = -1
            return -1000  

        return 50  # Reward for valid move

    def over(self):# check if game is over -1 for black win 1 for white win 0 for neither 
        black = False
        white = False
        #this loop lowkey cool so if it finds one and the other is found it returns 0 and at the end it checks for one returns based on that and if false returns other 
        for i in self.state[:64]:
            if i == 6:  
                white = True
                if black:
                    return 0
            elif i == -6: 
                black = True
                if white:
                    return 0
        if black:
            return -1 
        return 1
    #big optimization potential for is_valid_move 
    def is_valid_move(self, start, end):#if you need a comment to know what this code does im sorry your cooked 
        moves = self.generate_moves(start)
        return end in moves

    def generate_moves(self, start):  # Generates a list of all possible moves for a piece.
        moves = []
        piece = self.state[start]
        row, col = self.get_row_col(start)

        if piece == 1:  # White Pawn
            if self.is_valid_square(start - 8) and self.state[start - 8] == 0:
                moves.append(start - 8)
            # Double move from the starting position
            if row == 6 and self.state[start - 8] == 0 and self.state[start - 16] == 0:
                moves.append(start - 16)
            # Capture diagonally
            if self.is_valid_square(start - 9) and self.state[start - 9] < 0:  # Capture left
                moves.append(start - 9)
            if self.is_valid_square(start - 7) and self.state[start - 7] < 0:  # Capture right
                moves.append(start - 7)

        elif piece == 2:  # White Knight
            knight_moves = [-17, -15, -10, -6, 6, 10, 15, 17]  #possible moves 
            for move in knight_moves:
                new_index = start + move
                if self.is_valid_square(new_index) and self.state[new_index] != piece:
                    moves.append(new_index)

        elif piece == 3:  # White Bishop
            directions = [-9, -7, 7, 9]  # Diagonal directions
            for direction in directions:
                current_index = start
                while True:
                    current_index += direction
                    if not self.is_valid_square(current_index):
                        break
                    if self.state[current_index] != 0 and self.state[current_index] > 0:
                        break  # Blocked by another piece
                    moves.append(current_index)
                    if self.state[current_index] != 0:  # Piece in the way WOMP WOMP 
                        break

        elif piece == 4:  # White Rook
            directions = [-8, 8, -1, 1]  # Horizontal and vertical directions
            for direction in directions:
                current_index = start
                while True:
                    current_index += direction
                    if not self.is_valid_square(current_index):
                        break
                    if self.state[current_index] != 0 and self.state[current_index] > 0:
                        break  # Blocked by another piece
                    moves.append(current_index)
                    if self.state[current_index] != 0:  # Piece in the way
                        break

        elif piece == 5:  # White Queen
            directions = [-8, 8, -1, 1, -9, -7, 7, 9]  # All directions (rook + bishop)
            for direction in directions:
                current_index = start
                while True:
                    current_index += direction
                    if not self.is_valid_square(current_index):
                        break
                    if self.state[current_index] != 0 and self.state[current_index] > 0:
                        break  # Blocked by another piece
                    moves.append(current_index)
                    if self.state[current_index] != 0:  # Piece in the way
                        break

        elif piece == 6:  # White King
            directions = [-1, 1, -8, 8, -7, 7, -9, 9]  # All adjacent squares
            for direction in directions:
                new_index = start + direction
                if self.is_valid_square(new_index) and self.state[new_index] != piece:
                    moves.append(new_index)

        return moves
    #optimization potential remove the need to use get_row_col and to_index
    def get_row_col(self, index):#convert 1d index into 2d 
        return index // 8, index % 8
    def to_index(self, row, col):#converts 2d into 1d 
        return row * 8 + col

class coolmodel(nn.Module):#model time also cool model is a good name trust 
    def __init__(self, input_dim, output_dim):
        super(coolmodel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):#connects the lays to add more layers just repeat the pattern
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class ChessAgent:#chess Agent 
    def __init__(self, env, model, target_model, epsilon=.995, epsilon_min=0.1, epsilon_decay=0.99, gamma=0.99, learning_rate=0.010):
        self.env = env# variables needed for the model im not finna 
        self.model = model
        self.target_model = target_model
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)  # Store experiences in a deque
        self.batch_size = 64

    def epsilon_greedy_action(self, state):#selects move for greedy action 
        if random.random() < self.epsilon:
            start_pos = random.randint(0, 63)
            end_pos = random.randint(0, 63)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = self.model(state_tensor)
                start_pos = torch.argmax(q_values[:, :64]).item()
                end_pos = torch.argmax(q_values[:, 64:]).item()
        return start_pos, end_pos
    
    def store_experience(self, state, action_start, action_end, reward, next_state, done):#stores experience ... thats it 
        # Convert state and next_state to numpy arrays before storing
        self.memory.append((np.array(state), action_start, action_end, reward, np.array(next_state), done))
    
    def sample_batch(self):
        batch = random.sample(self.memory, self.batch_size)
        
        # Convert the batch into numpy arrays efficiently
        states, actions_start, actions_end, rewards, next_states, dones = zip(*batch)
        
        # Convert the list of numpy arrays into a single numpy array for faster tensor conversion I think
        #I dont rly know what im doing with  but I think this is faster 
        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        actions_start = np.array(actions_start, dtype=np.int64)
        actions_end = np.array(actions_end, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        return states, actions_start, actions_end, rewards, next_states, dones

    def train(self):#trains the bot 
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch of experiences
        states, actions_start, actions_end, rewards, next_states, dones = self.sample_batch()
        
        # Convert numpy arrays to PyTorch tensors at once
        states_tensor = torch.tensor(states)
        next_states_tensor = torch.tensor(next_states)
        actions_start_tensor = torch.tensor(actions_start)
        actions_end_tensor = torch.tensor(actions_end)
        rewards_tensor = torch.tensor(rewards)
        dones_tensor = torch.tensor(dones)

        next_q_values = self.target_model(next_states_tensor)
        
        # Calculate target Q-values for the start and end positions
        target_start_q_values = rewards_tensor + (self.gamma * torch.max(next_q_values[:, 64:], dim=1)[0] * (1 - dones_tensor))
        target_end_q_values = rewards_tensor + (self.gamma * torch.max(next_q_values[:, :64], dim=1)[0] * (1 - dones_tensor))
        
        # Get the current Q-values from the model
        q_values = self.model(states_tensor)
        
        # Select the Q-values for the actions taken
        new_start_qvals = q_values.gather(1, actions_start_tensor.unsqueeze(1))
        new_end_qvalues = q_values.gather(1, actions_end_tensor.unsqueeze(1))
        
        # Compute the loss for start and end positions
        loss_start = nn.functional.mse_loss(new_start_qvals, target_start_q_values.unsqueeze(1))
        loss_end = nn.functional.mse_loss(new_end_qvalues, target_end_q_values.unsqueeze(1))
        
        loss = loss_start + loss_end
        
        # Perform the optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self):#updates the model
        self.target_model.load_state_dict(self.model.state_dict())

# Initialize environment and agents
env = ChessEnv()
model_white = coolmodel(input_dim=64, output_dim=128)
model_black = coolmodel(input_dim=64, output_dim=128)
target_model_white = coolmodel(input_dim=64, output_dim=128)
target_model_black = coolmodel(input_dim=64, output_dim=128)

target_model_white.load_state_dict(model_white.state_dict())
target_model_black.load_state_dict(model_black.state_dict())

agent_white = ChessAgent(env, model_white, target_model_white)
agent_black = ChessAgent(env, model_black, target_model_black)

# Training loop for both agents
num_episodes = 10000
time = dt.now()

for episode in range(num_episodes):
    gc.collect()
    env.reset()
    state = np.array(env.state[:64])
    total_reward_white = 0
    total_reward_black = 0
    done = False
    
    while not done:
        start_pos, end_pos = agent_white.epsilon_greedy_action(state)
        reward_white = env.move(start_pos, end_pos)
        next_state = np.array(env.state[:64])
        done = env.over()
        
        # Store experience and train white agent
        agent_white.store_experience(state, start_pos, end_pos, reward_white, next_state, done)
        agent_white.train()
        
        # Black's turn 
        if not done:
            env.flip()
            start_pos, end_pos = agent_black.epsilon_greedy_action(next_state)
            reward_black = env.move(start_pos, end_pos)
            next_state_black = np.array(env.state[:64])
            done = env.over()
            
            # Store experience and train black agent
            agent_black.store_experience(next_state, start_pos, end_pos, reward_black, next_state_black, done)
            agent_black.train()
        
        state = next_state
        total_reward_white += reward_white
        total_reward_black += reward_black

    # Decay epsilon for both agents
    if agent_white.epsilon > agent_white.epsilon_min:
        agent_white.epsilon *= agent_white.epsilon_decay
    if agent_black.epsilon > agent_black.epsilon_min:
        agent_black.epsilon *= agent_black.epsilon_decay

    # Periodically update the target networks, remove the time stuff if u want it to run faster but its kinda cool seing time delta
    if episode % 10 == 0:
        agent_white.update_target_network()
        agent_black.update_target_network()
        print(f"Episode {episode}, White Epsilon {agent_white.epsilon}, Black Epsilon {agent_black.epsilon}")
        print(f"Runtime of last 10 episodes: {dt.now() - time}")
        time = dt.now()
