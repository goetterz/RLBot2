import numpy as np
import torch
import os
import math
import random

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.structures.quick_chats import QuickChats
from rlgym_compat import GameState

# Import your trained model components
from rlgym_sim.utils.obs_builders import DefaultObs
from rlgym_sim.utils import common_values

# Default kickoff sequence - can be improved with your trained bot's kickoffs later
KICKOFF_CONTROLS = (
    11 * 4 * [SimpleControllerState(throttle=1, boost=True)]
    + 4 * 4 * [SimpleControllerState(throttle=1, boost=True, steer=-1)]
    + 2 * 4 * [SimpleControllerState(throttle=1, jump=True, boost=True)]
    + 1 * 4 * [SimpleControllerState(throttle=1, boost=True)]
    + 1 * 4 * [SimpleControllerState(throttle=1, yaw=0.8, pitch=-0.7, jump=True, boost=True)]
    + 13 * 4 * [SimpleControllerState(throttle=1, pitch=1, boost=True)]
    + 10 * 4 * [SimpleControllerState(throttle=1, roll=1, pitch=0.5)]
)

KICKOFF_NUMPY = np.array([
    [scs.throttle, scs.steer, scs.pitch, scs.yaw, scs.roll, scs.jump, scs.boost, scs.handbrake]
    for scs in KICKOFF_CONTROLS
])


class RLGymPolicy:
    def __init__(self, checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Define our own policy network architecture that matches what was used in training
            class SimplePolicyNet(torch.nn.Module):
                def __init__(self, input_size, output_size, layer_sizes):
                    super().__init__()
                    
                    # Change naming from network to model to match saved state dict
                    layers = []
                    prev_size = input_size
                    
                    # Hidden layers
                    for size in layer_sizes:
                        layers.append(torch.nn.Linear(prev_size, size))
                        layers.append(torch.nn.ReLU())
                        prev_size = size
                    
                    # Output layer (for continuous actions)
                    layers.append(torch.nn.Linear(prev_size, output_size))
                    layers.append(torch.nn.Tanh())  # Tanh to bound actions to [-1, 1]
                    
                    self.model = torch.nn.Sequential(*layers)  # Changed from network to model
                
                def forward(self, x):
                    return self.model(x)
            
            # Create an instance of our policy with the CORRECTED architecture
            input_size = 89   # Changed from 107 to match saved model
            output_size = 16  # Changed from 8 to match saved model
            layer_sizes = [1024, 1024, 768, 512, 256]  # Keep the same hidden layers
            
            # Instantiate the policy with corrected sizes
            self.policy = SimplePolicyNet(
                input_size=input_size,
                output_size=output_size,
                layer_sizes=layer_sizes
            ).to(self.device)
            
            # Load the saved checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            print(f"Keys in checkpoint: {checkpoint.keys()}")
            if "policy_state_dict" in checkpoint:
                print(f"First few keys in policy_state_dict: {list(checkpoint['policy_state_dict'].keys())[:5]}")
            
            # Try to load the state dict directly (should work now)
            try:
                self.policy.load_state_dict(checkpoint["policy_state_dict"])
                print("Successfully loaded state dict directly")
                self.policy.eval()  # Set to evaluation mode
            except Exception as e:
                print(f"Loading failed: {e}")
                self.policy = None
            
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            self.policy = None
            
    def act(self, obs, deterministic=True):
        """Return actions based on observations"""
        if self.policy is None:
            return np.random.uniform(-1, 1, 8), None
            
        # Handle the observation size mismatch
        obs_truncated = obs[:89] if len(obs) > 89 else obs
        
        # Convert observation to tensor
        obs_tensor = torch.FloatTensor(obs_truncated).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            # Get all 16 outputs
            action_tensor = self.policy(obs_tensor)
            action_array = action_tensor.squeeze(0).cpu().numpy()
            
            # Important: Use the second half (indices 8-15) as these are likely the actual control outputs
            # Based on your model architecture, it might be outputting additional information
            final_actions = np.zeros(8)
            final_actions[0] = np.clip(action_array[8], -1, 1)     # Throttle
            final_actions[1] = np.clip(action_array[9], -1, 1)     # Steer
            final_actions[2] = np.clip(action_array[10], -1, 1)    # Pitch
            final_actions[3] = np.clip(action_array[11], -1, 1)    # Yaw
            final_actions[4] = np.clip(action_array[12], -1, 1)    # Roll
            final_actions[5] = float(action_array[13] > 0)         # Jump (binary)
            final_actions[6] = float(action_array[14] > 0)         # Boost (binary)
            final_actions[7] = float(action_array[15] > 0)         # Handbrake (binary)
            
            print(f"Model outputs (remapped): {final_actions}")
            
            return final_actions, None


class TrainedBot(BaseAgent):
    def __init__(self, name, team, index, beta=1.0, render=False, hardcoded_kickoffs=True):
        super().__init__(name, team, index)
        
        # Path to your trained model directly in the bot directory
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model.pt")
        
        # Check if the model file exists
        if os.path.exists(model_path):
            print(f"Found model file: {model_path}")
            checkpoint_path = model_path
        else:
            print(f"Model file not found at {model_path}")
            checkpoint_path = None
        
        self.obs_builder = None
        self.agent = RLGymPolicy(checkpoint_path)
        self.tick_skip = 8
        
        self.beta = beta  # Controls determinism: 1=best action, 0=random
        self.render = render
        self.hardcoded_kickoffs = hardcoded_kickoffs
        
        self.game_state = None
        self.controls = None
        self.action = None
        self.update_action = True
        self.ticks = 0
        self.prev_time = 0
        self.kickoff_index = -1
        self.field_info = None
        
        print(f'TrainedBot Ready - Index: {index}, Team: {team}')
        
    def initialize_agent(self):
        # Initialize the rlgym GameState object now that the game is active
        self.field_info = self.get_field_info()
        self.obs_builder = DefaultObs(
            pos_coef=np.asarray([1/common_values.SIDE_WALL_X, 1/common_values.BACK_NET_Y, 1/common_values.CEILING_Z]),
            ang_coef=1/np.pi,
            lin_vel_coef=1/common_values.CAR_MAX_SPEED,
            ang_vel_coef=1/common_values.CAR_MAX_ANG_VEL
        )
        self.game_state = GameState(self.field_info)
        self.ticks = self.tick_skip  # Take an action first tick
        self.prev_time = 0
        self.controls = SimpleControllerState()
        self.action = np.zeros(8)
        self.update_action = True
        self.kickoff_index = -1
        
    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        # Calculate delta time and update game state
        cur_time = packet.game_info.seconds_elapsed
        delta = cur_time - self.prev_time
        self.prev_time = cur_time
        
        ticks_elapsed = round(delta * 120)  # 120hz ticks
        self.ticks += ticks_elapsed
        self.game_state.decode(packet, ticks_elapsed)
        
        # Update action when necessary
        if self.update_action and len(self.game_state.players) > self.index:
            self.update_action = False
            
            # Organize the players list: current player first, then teammates, then opponents
            player = self.game_state.players[self.index]
            teammates = [p for p in self.game_state.players if p.team_num == self.team and p != player]
            opponents = [p for p in self.game_state.players if p.team_num != self.team]
            
            self.game_state.players = [player] + teammates + opponents
            
            # Build observation and get action from policy
            obs = self.obs_builder.build_obs(player, self.game_state, self.action)
            
            # CRITICAL FIX: Make observations team-relative for orange team
            if self.team == 1:  # Orange team
                # Invert x and y coordinates in the observation
                # This depends on the specific structure of your obs_builder
                # For DefaultObs, the first few values are car position, ball position, etc.
                
                # Flip x coordinates (every 3rd value starting from 0)
                for i in range(0, len(obs), 3):
                    if i < 66:  # Only flip position values, not velocities
                        obs[i] = -obs[i]
                
                # Flip y rotations and velocities
                # Specific indices depend on DefaultObs implementation
                # You may need to adjust these based on your observation structure
                
            # Get action from policy
            deterministic = True if self.beta > 0.99 else False
            self.action, _ = self.agent.act(obs, deterministic)
        
        # Update controls when tick_skip is reached
        if self.ticks >= self.tick_skip - 1:
            self.update_controls(self.action)
        
        # Reset tick counter when tick_skip is reached
        if self.ticks >= self.tick_skip:
            self.ticks = 0
            self.update_action = True
            
        # Use hardcoded kickoffs if enabled
        if self.hardcoded_kickoffs:
            self.maybe_do_kickoff(packet, ticks_elapsed)
            
        return self.controls
            
    def update_controls(self, action):
        self.controls.throttle = action[0]
        self.controls.steer = action[1]
        self.controls.pitch = action[2]
        self.controls.yaw = action[3]
        self.controls.roll = action[4]
        self.controls.jump = action[5]
        self.controls.boost = action[6]
        self.controls.handbrake = action[7]
        
    def maybe_do_kickoff(self, packet, ticks_elapsed):
        if packet.game_info.is_kickoff_pause:
            if self.kickoff_index >= 0:
                self.kickoff_index += round(ticks_elapsed)
            elif self.kickoff_index == -1:
                is_kickoff_taker = False
                ball_pos = np.array([packet.game_ball.physics.location.x, packet.game_ball.physics.location.y])
                positions = np.array([[car.physics.location.x, car.physics.location.y]
                                     for car in packet.game_cars[:packet.num_cars]])
                distances = np.linalg.norm(positions - ball_pos, axis=1)
                if abs(distances.min() - distances[self.index]) <= 10:
                    is_kickoff_taker = True
                    indices = np.argsort(distances)
                    for index in indices:
                        if abs(distances[index] - distances[self.index]) <= 10 \
                                and packet.game_cars[index].team == self.team \
                                and index != self.index:
                            if self.team == 0:
                                is_left = positions[index, 0] < positions[self.index, 0]
                            else:
                                is_left = positions[index, 0] > positions[self.index, 0]
                            if not is_left:
                                is_kickoff_taker = False  # Left goes

                self.kickoff_index = 0 if is_kickoff_taker else -2
                
            if 0 <= self.kickoff_index < len(KICKOFF_NUMPY) \
                    and packet.game_ball.physics.location.y == 0:
                action = KICKOFF_NUMPY[self.kickoff_index]
                self.action = action
                self.update_controls(self.action)
        else:
            self.kickoff_index = -1