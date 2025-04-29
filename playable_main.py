import time
import flappy_bird_gym
import keyboard  # To capture keyboard input

# Create the environment
env = flappy_bird_gym.make("FlappyBird-rgb-v0")

# Reset the environment
obs = env.reset()

# Define the keys for actions
JUMP_KEY = 'space'  # Space key will make the bird jump

score = 0

while True:
    # Check for user input
    action = 0  # Default action: do nothing

    # If the space key is pressed, make the bird jump
    if keyboard.is_pressed(JUMP_KEY):
        action = 1  # Action 1 corresponds to jump

    # Process the environment with the chosen action
    obs, reward, done, info = env.step(action)
    log = f"Observation: {obs.shape}, Reward: {reward}, Done: {done}, Info: {info}"
    print(log)

    # Rendering the game
    env.render()
    time.sleep(1 / 30)  # 60 FPS

    if (info['score'] > score):
        score = info['score']
        done = False

    # Check if the game is over
    if done:
        break

# Close the environment when done
env.close()
