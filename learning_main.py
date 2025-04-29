import flappy_bird_gym
import q_learn
import log
import constants

class FlappyBirdAgent:
    def __init__(self, env):
        self.env = env

        self.best_score = constants.INITIAL_BEST_SCORE
        log.log(f"Initial best score: {self.best_score}", console_output=True)

        self.consecutive_3s = constants.INITIAL_CONSECUTIVE_3S
        log.log(f"Initial consecutive 3s: {self.consecutive_3s}", console_output=True)

    def run_episode(self, episode_num):
        # log.episode_label(f"Episode {episode_num} started - Current best score: {self.best_score}", console_output=True)
        current_state = self.env.reset()
        total_reward = 0
        episode_score = 0
        number_of_ups = 0
        number_of_downs = 0
        number_of_ups_explored = 0
        number_of_downs_explored = 0
        number_of_ups_exploited = 0
        number_of_downs_exploited = 0
        log.episode_data(
            f"Episode reward: {total_reward} - "
            f"Episode score: {episode_score} - "
            f"Number of ups: {number_of_ups} - "
            f"Number of downs: {number_of_downs} - "
            f"Number of ups explored: {number_of_ups_explored} - "
            f"Number of downs explored: {number_of_downs_explored} - "
            f"Number of ups exploited: {number_of_ups_exploited} - "
            f"Number of downs exploited: {number_of_downs_exploited}",
            console_output=False
        )
        
        while True:
            action, type = q_learn.get_action(current_state, episode_score)
            if action == 1:
                number_of_ups += 1
                if type == 'exploration':
                    number_of_ups_explored += 1
                elif type == 'exploitation':
                    number_of_ups_exploited += 1
                else:
                    raise ValueError(f"Unknown action type: {type}")
            else:
                number_of_downs += 1
                if type == 'exploration':
                    number_of_downs_explored += 1
                elif type == 'exploitation':
                    number_of_downs_exploited += 1
                else:
                    raise ValueError(f"Unknown action type: {type}")
                
            next_state, _, end_game, info = self.env.step(action)

            action_score = info['score']
            reward = self._compute_reward(episode_score, action_score, end_game)

            episode_score = action_score
            # log.episode_data(f"Action: {action} - Reward: {reward} - end_game: {end_game}", console_output=False)

            total_reward += reward
            q_learn.add_to_replay_buffer(current_state, action, next_state, reward, end_game)

            if end_game:
                break
            
            current_state = next_state
                
        q_learn.episode_update(episode_num, episode_score)
        self._update_best_score(episode_score)
    
        log.episode_label(
            f"End of episode {episode_num}", console_output=True)
        
        log.episode_data(
            f"Current Score: {episode_score} - "
            f"Total Reward: {total_reward} - "
            f"Number of ups: {number_of_ups} - "
            f"Number of downs: {number_of_downs}",
            console_output=True
        )

        log.episode_data(
            f"Number of ups explored: {number_of_ups_explored} - "
            f"Number of downs explored: {number_of_downs_explored} - "
            f"Number of ups exploited: {number_of_ups_exploited} - "
            f"Number of downs exploited: {number_of_downs_exploited}",
            console_output=True
        )

        if(episode_num % constants.RESUME_SAVE_RATE == 0):
            log.log(f"Current best score: {self.best_score} - Consecutive 3s: {self.consecutive_3s}", console_output=True)

    def _compute_reward(self, current_score, action_score, end_game):
        """Compute reward and step in the environment."""
        if end_game:
            return -1.5
        elif action_score > current_score:
            return 2
        return 1

    def _update_best_score(self, episode_score):
        """Update the best score if the current episode score is higher."""
        if episode_score > self.best_score:
            self.best_score = episode_score
            q_learn.save_model(constants.PATH_TO_BEST_MODEL)
            log.episode_data(f"Best model saved - best score {self.best_score}", console_output=True)
        
        if self.best_score >= 3:
            self.consecutive_3s += 1
        else:
            self.consecutive_3s = 0

    def train_by_number_of_episodes(self, num_episodes):
        log.log("Running training by number of episodes", console_output=True)
        log.log(f"Number of episodes necessary to stop: {num_episodes}", console_output=True)
        for episode_num in range(1, num_episodes + 1):
            self.run_episode(episode_num)
    
    def train_by_consecutive_3s(self):
        log.log("Running training by consecutive 3s", console_output=True)
        log.log(f"Consecutive 3s necessary to stop: {constants.MAX_CONSECUTIVE_3S}", console_output=True)

        episode = 1
        while self.consecutive_3s < constants.MAX_CONSECUTIVE_3S:
            self.run_episode(episode)
            episode += 1

def main():
    log.start_learn()
    log.log_constants()

    # Create environment and agent
    env_name = "FlappyBird-rgb-v0"
    env = flappy_bird_gym.make(env_name)
    log.log(f"Environment: {env_name}", console_output=True)

    agent = FlappyBirdAgent(env)

    # Run training
    # agent.train_by_number_of_episodes(constants.EPISODES)
    agent.train_by_consecutive_3s()

    # Close the environment when done
    env.close()
    log.end_learn()

if __name__ == "__main__":
    main()
