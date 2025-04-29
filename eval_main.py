import flappy_bird_gym
import q_learn;
import time

# Create the environment
env = flappy_bird_gym.make("FlappyBird-rgb-v0")

obs = env.reset()
current_state, reward, end_game, info = env.step(0)

score = 0


while True:
    action = q_learn.get_action(current_state);
    next_state, reward, end_game, info = env.step(action)
    current_state = next_state

    if(info['score'] > score):
        score = info['score']
    
    env.render()
    time.sleep(1 / 45)

    if end_game:
        break

env.close()
print(f"GAME OVER - Score: {score}")