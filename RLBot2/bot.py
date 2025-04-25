import numpy as np
from rlgym_sim.utils.gamestates import GameState
from rlgym_sim.utils.reward_functions import RewardFunction
from rlgym_sim.utils import common_values
from rlgym_ppo.util import MetricsLogger


class SpeedTowardBallReward(RewardFunction):
    def __init__(self):
        super().__init__()
    
    def reset(self, initial_state: GameState):
        pass
        
    def get_reward(self, player, state: GameState, previous_action: np.ndarray) -> float:
        car_pos = player.car_data.position
        ball_pos = state.ball.position
        
        car_vel = player.car_data.linear_velocity
        
        pos_diff = ball_pos - car_pos
        # Fix: Use numpy's built-in normalization instead of math.vec_normalize
        dist = np.linalg.norm(pos_diff)
        if dist == 0:
            return 0.0  # Avoid division by zero
        norm_pos_diff = pos_diff / dist
        
        vel_mag = np.linalg.norm(car_vel)
        if vel_mag == 0:
            return 0.0
        norm_vel = car_vel / vel_mag
        
        # Dot product to see how aligned velocity is with direction to ball
        alignment = np.dot(norm_pos_diff, norm_vel)
        
        # Return positive reward only when moving toward the ball
        if alignment > 0:
            return float(alignment * vel_mag / common_values.CAR_MAX_SPEED)
        return 0.0


class FaceBallReward(RewardFunction):
    def __init__(self):
        super().__init__()
    
    def reset(self, initial_state: GameState):
        pass
        
    def get_reward(self, player, state: GameState, previous_action: np.ndarray) -> float:
        car_pos = player.car_data.position
        ball_pos = state.ball.position
        
        car_forward = player.car_data.forward()
        
        pos_diff = ball_pos - car_pos
        # Fix: Use numpy's built-in normalization
        dist = np.linalg.norm(pos_diff)
        if dist == 0:
            return 0.0  # Avoid division by zero
        norm_pos_diff = pos_diff / dist
        
        # Dot product to see how aligned forward direction is with direction to ball
        return float(np.dot(norm_pos_diff, car_forward))


class AirReward(RewardFunction):
    def __init__(self):
        super().__init__()
    
    def reset(self, initial_state: GameState):
        pass
        
    def get_reward(self, player, state: GameState, previous_action: np.ndarray) -> float:
        # Check if player is in the air
        return float(not player.on_ground)


class VelocityBallToGoalReward(RewardFunction):
    def __init__(self):
        super().__init__()
        
    def reset(self, initial_state: GameState):
        pass
        
    def get_reward(self, player, state: GameState, previous_action: np.ndarray) -> float:
        # Get the ball's velocity
        ball_vel = state.ball.linear_velocity
        
        # Get the target goal's position based on player's team
        if player.team_num == 0:  # Blue team - target orange goal (positive y)
            goal_direction = np.array([0, 1, 0])
        else:  # Orange team - target blue goal (negative y)
            goal_direction = np.array([0, -1, 0])
        
        # Compute the component of the ball's velocity toward the goal
        vel_toward_goal = np.dot(ball_vel, goal_direction)
        
        # Normalize by max ball speed to keep rewards in a reasonable range
        return float(vel_toward_goal / common_values.BALL_MAX_SPEED)


class ExampleLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        return [game_state.players[0].car_data.linear_velocity,
                game_state.players[0].car_data.rotation_mtx(),
                game_state.orange_score]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_linvel = np.zeros(3)
        for metric_array in collected_metrics:
            p0_linear_velocity = metric_array[0]
            avg_linvel += p0_linear_velocity
        avg_linvel /= len(collected_metrics)
        report = {"x_vel":avg_linvel[0],
                  "y_vel":avg_linvel[1],
                  "z_vel":avg_linvel[2],
                  "Cumulative Timesteps":cumulative_timesteps}
        wandb_run.log(report)


def build_rocketsim_env():
    import rlgym_sim
    from rlgym_sim.utils.reward_functions import CombinedReward
    from rlgym_sim.utils.reward_functions.common_rewards import EventReward
    from rlgym_sim.utils.obs_builders import DefaultObs
    from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
    from rlgym_sim.utils import common_values
    from rlgym_sim.utils.action_parsers import ContinuousAction
    from rlgym_sim.utils.state_setters import RandomState

    spawn_opponents = True
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 10
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = ContinuousAction()
    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]

    # Early-stage focused rewards using the from_zipped format
    reward_fn = CombinedReward.from_zipped(
        # Format is (func, weight)
        (EventReward(touch=0.25, team_goal=1, concede=-1, demo=0), 20.0),  # Reduced touch reward, added moderate goal rewards
        (VelocityBallToGoalReward(), 3.0),  # Reward for moving the ball toward the goal
        (SpeedTowardBallReward(), 1.0),  # Reduced relative to VelocityBallToGoalReward
        (FaceBallReward(), 0.5),  # Face the ball to avoid backward driving
        (AirReward(), 0.25)  # Small encouragement for aerials
    )

    # Create RandomState state setter with recommended settings
    state_setter = RandomState(
        ball_rand_speed=True,    # Random ball velocity
        cars_rand_speed=True,    # Random car velocity
        cars_on_ground=False     # Allow cars to spawn in the air
    )

    obs_builder = DefaultObs(
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL)

    env = rlgym_sim.make(tick_skip=tick_skip,
                         team_size=team_size,
                         spawn_opponents=spawn_opponents,
                         terminal_conditions=terminal_conditions,
                         reward_fn=reward_fn,
                         obs_builder=obs_builder,
                         action_parser=action_parser,
                         state_setter=state_setter)  # Add state setter here

    return env

import os

if __name__ == "__main__":
    from rlgym_ppo import Learner
    metrics_logger = ExampleLogger()
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = "data/checkpoints/rlgym-ppo-run"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Uncomment to load the most recent checkpoint
    # if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
    #     latest_checkpoint_dir = checkpoint_dir + "/" + str(max(os.listdir(checkpoint_dir), key=lambda d: int(d)))
    # else:
    #     latest_checkpoint_dir = None

    # V100-optimized settings
    n_proc = 32  # Increased processes for faster data collection
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(build_rocketsim_env,
                  n_proc=n_proc,
                  min_inference_size=min_inference_size,
                  metrics_logger=metrics_logger,
                  render=False,
                  ppo_batch_size=100_000,  # Double the original batch size
                  policy_layer_sizes=[1024, 1024, 768, 512, 256],  # Keep original large network
                  critic_layer_sizes=[1024, 1024, 768, 512, 256],  # Keep original large network
                  ts_per_iteration=100_000,  # Match with batch size
                  exp_buffer_size=300_000,  # Double the original buffer size
                  ppo_minibatch_size=50_000,  # Smaller division for more gradient updates
                  ppo_ent_coef=0.001,
                  ppo_epochs=3,  # Increased for better convergence
                  policy_lr=2e-4,
                  critic_lr=2e-4,
                  standardize_returns=True,
                  standardize_obs=True,
                  save_every_ts=500_000,
                  checkpoints_save_folder=checkpoint_dir,
                  # checkpoint_load_folder=latest_checkpoint_dir,  # Uncomment to load checkpoint
                  add_unix_timestamp=False,
                  timestep_limit=200_000_000,  # Doubled training limit
                  log_to_wandb=False)
    learner.learn()