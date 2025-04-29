import numpy as np
import cv2
import dql_model
import torch
from collections import deque
import log
import constants

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.log(f"Device used for computations: {device}", console_output=True)

__q_model = None
__optimizer = None
__replay_buffer = None

__exploration_epsilon_p1 = None
__exploration_epsilon_p2 = None
__exploration_epsilon_p3 = None

def __initialize_q_learn():
    global __q_model, __optimizer, __replay_buffer, __exploration_epsilon_p1, __exploration_epsilon_p2, __exploration_epsilon_p3, __target_model_update_rate
    
    __q_model = dql_model.DQLModel().to(device)

    __replay_buffer = deque(maxlen=constants.REPLAY_BUFFER_SIZE)
    log.log(f"Replay buffer size: {constants.REPLAY_BUFFER_SIZE}", console_output=True)

    __optimizer = torch.optim.Adam(__q_model.parameters(), lr=constants.LEARNING_RATE)
    log.log(f"Optimizer: Adam - Learning rate: {constants.LEARNING_RATE}", console_output=True)

    __exploration_epsilon_p1 = constants.INITIAL_EPSILON_P1
    __exploration_epsilon_p2 = constants.INITIAL_EPSILON_P2
    __exploration_epsilon_p3 = constants.INITIAL_EPSILON_P3
    log.log(f"Exploration epsilon - p1: {__exploration_epsilon_p1} - p2: {__exploration_epsilon_p2} - p3: {__exploration_epsilon_p3}", console_output=True)

log.log(f'Model scope: {constants.MODEL_SCOPE}', console_output=True)

if(constants.MODEL_SCOPE == 'zero-shot-train'):
    log.log('SCRATCH training model')

    __initialize_q_learn()

    __q_model.train()

elif(constants.MODEL_SCOPE == 'weighted-shot-train'):
    log.log(f'Using model weights from {constants.PATH_TO_BEST_MODEL} to train', console_output=True)

    __initialize_q_learn()
    checkpoint = torch.load(constants.PATH_TO_BEST_MODEL, weights_only=True)

    __q_model.load_state_dict(checkpoint['q_model'])
    __optimizer.load_state_dict(checkpoint['optimizer'])

    __q_model.train()

elif(constants.MODEL_SCOPE == 'eval'):
    log.log(f'MODEL SET TO EVALUATION MODE - Model used: {constants.PATH_TO_BEST_MODEL}', console_output=True)

    __q_model = dql_model.DQLModel().to(device)

    checkpoint = torch.load(constants.PATH_TO_BEST_MODEL, weights_only=True)    

    __q_model.load_state_dict(checkpoint['q_model'])

    __exploration_epsilon_p1 = constants.MIN_EPSILON

    __q_model.eval()
else:
    raise ValueError(f"Invalid model scope: {constants.MODEL_SCOPE}")

def get_action(current_state, current_score):
    modified_frame = __modify_frame(current_state)
    torch_frame = torch.tensor(modified_frame, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    epsilon_in_use = None
    if(current_score < 1):
        epsilon_in_use = __exploration_epsilon_p1
    elif(current_score < 2):
        epsilon_in_use = __exploration_epsilon_p2
    else:
        epsilon_in_use = __exploration_epsilon_p3

    if np.random.rand() < epsilon_in_use:
        choice = np.random.choice([1, 0], p=[constants.UPWARD_MOVEMENT_EXPLORATION_PICK_RATE, 1 - constants.UPWARD_MOVEMENT_EXPLORATION_PICK_RATE])
        # log.episode_data(f'Went for exploration - Chose: {choice}', console_output=False)
        return choice, 'exploration'
    # log.episode_data('Went for exploitation', console_output=False)
    
    with torch.no_grad():
        q_values = __q_model(torch_frame)
    # log.episode_data(f'Q values: {q_values}', console_output=False)
    choice = torch.argmax(q_values).item()
    
    return choice, 'exploitation'


def add_to_replay_buffer(current_state, action, next_state, reward, end_found):
    current_state_modified = __modify_frame(current_state)
    next_state_modified = __modify_frame(next_state)

    __replay_buffer.append((current_state_modified, action, next_state_modified, reward, end_found))
    # log.episode_data(f"Replay buffer size: {len(__replay_buffer)}", console_output=False)

def episode_update(episode, current_score):

    __train_on_replay_buffer_a_number_of_times()

    global __exploration_epsilon_p1, __exploration_epsilon_p2, __exploration_epsilon_p3, __target_model_update_rate
    if (current_score < 1):
        __exploration_epsilon_p1 = max(constants.MIN_EPSILON, __exploration_epsilon_p1 * constants.EPSILON_UPDATE_RATE)
        log.episode_data(f"Exploration epsilon p1: {__exploration_epsilon_p1}", console_output=True)
    elif (current_score < 2):
        __exploration_epsilon_p2 = max(constants.MIN_EPSILON, __exploration_epsilon_p2 * constants.EPSILON_UPDATE_RATE)
        log.episode_data(f"Exploration epsilon p2: {__exploration_epsilon_p2}", console_output=True)
        __exploration_epsilon_p1 = max(constants.MIN_EPSILON, __exploration_epsilon_p1 * constants.EPSILON_UPDATE_RATE)
        log.episode_data(f"Exploration epsilon p1: {__exploration_epsilon_p1}", console_output=True)
    else:
        __exploration_epsilon_p3 = max(constants.MIN_EPSILON, __exploration_epsilon_p3 * constants.EPSILON_UPDATE_RATE)
        log.episode_data(f"Exploration epsilon p3: {__exploration_epsilon_p3}", console_output=True)
        __exploration_epsilon_p2 = max(constants.MIN_EPSILON, __exploration_epsilon_p2 * constants.EPSILON_UPDATE_RATE)
        log.episode_data(f"Exploration epsilon p2: {__exploration_epsilon_p2}", console_output=True)
        __exploration_epsilon_p1 = max(constants.MIN_EPSILON, __exploration_epsilon_p1 * constants.EPSILON_UPDATE_RATE)
        log.episode_data(f"Exploration epsilon p1: {__exploration_epsilon_p1}", console_output=True)


    if(episode%constants.RESUME_SAVE_RATE == 0): 
        save_model(constants.PATH_TO_RESUME_MODEL)
        log.log("Resume model saved", console_output=True)
       

def __train_on_replay_buffer_a_number_of_times():
    total_loss = 0.0
    for i in range(constants.REPLAY_GAME_EPOCHS):
        total_loss += __train_on_replay_buffer()

    log.episode_data(f"Total loss: {total_loss}", console_output=True)   

def __train_on_replay_buffer():

    total_loss = 0.0
    last_replay_buffer_index = len(__replay_buffer) - 1

    for i in range(last_replay_buffer_index, -1, -1):
        loss = __train_on_frame(__replay_buffer[i])
        total_loss += loss
    
    log.episode_data(f"Total loss: {total_loss}", console_output=False)
    return total_loss

def __train_on_frame(frame):

    if(np.random.rand() > constants.REPLAY_CHANCE):
        return 0.0

    current_state, action, next_state, reward, end_found = frame

    current_state_tensor = torch.tensor(current_state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    q_values = __q_model(current_state_tensor)

    next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    next_q_values = __q_model(next_state_tensor)
    max_next_q_value = torch.max(next_q_values).item()

    target = reward + constants.GAMMA * max_next_q_value * (1 - end_found)

    loss = torch.nn.functional.mse_loss(q_values[0][action], torch.tensor(target, dtype=torch.float32, device=device))
    
    __optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(__q_model.parameters(), max_norm=constants.GRADIENT_CLIP_VALUE)
    __optimizer.step()

    return loss.item()

def save_model(path):
    try:
        torch.save({
            'q_model': __q_model.state_dict(),
            'optimizer': __optimizer.state_dict()
        }, path)
        log.log(f"Model saved to {path}")
    except Exception as e:
        log.log(f"Failed to save model to {path}: {e}")

def _show_tensor_values(tensor):
    for i, value in enumerate(tensor.flatten()):
        print(f"Value {i}: {value}")

def __modify_frame(frame):
    """
    Processes a 3D image array:
    1. Validates the input shape (288, 512, 3).
    2. Converts the image to grayscale.
    3. Resizes the grayscale image to (128, 72).

    Parameters:
    - image_array (numpy.ndarray): Input 3D array of shape (288, 512, 3).

    Returns:
    - numpy.ndarray: Resized grayscale image of shape (128, 72).
    """
    # Validate input shape
    if frame.shape != (288, 512, 3):
        raise ValueError(f"Input array must have dimensions (288, 512, 3), but got {frame.shape}")

    # Convert to grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the image to 72x128 pixels
    resized_image = cv2.resize(gray_image, (128, 72))

    # _show_image(resized_image)

    return resized_image / 255.0

def _show_image(image_array):
    """
    Displays an image using OpenCV.

    Parameters:
    - image_array (numpy.ndarray): Input image array.
    """
    print(f"Image shape: {image_array.shape}")
    cv2.imshow('Image', image_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()