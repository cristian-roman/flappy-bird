import os
import sys
from datetime import datetime
import constants

def episode_label(message, console_output):
    log(f'\t{message}', console_output)
   
def episode_data(message, console_output):
    log(f'\t\t{message}', console_output)

def start_learn():
    log('==============================================================================================', console_output=True)
    log('Start of training session', console_output=True)

def end_learn():
    log('End of training session', console_output=True)
    log('==============================================================================================', console_output=True)

def log_constants():
    log(f'Minimum exploration epsilon: {constants.MIN_EPSILON}', console_output=True)
    log(f'Upward movement exploration pick rate: {constants.UPWARD_MOVEMENT_EXPLORATION_PICK_RATE}', console_output=True)
    log(f'Gradient clip value: {constants.GRADIENT_CLIP_VALUE}', console_output=True)
    log(f'Epsilon update rate: {constants.EPSILON_UPDATE_RATE}', console_output=True)
    log(f'Gamma: {constants.GAMMA}', console_output=True)
    log(f'Resume save rate: {constants.RESUME_SAVE_RATE}', console_output=True)
    
def log(message, console_output=False):
    __create_log_file_if_inexistent()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open('log.txt', 'a') as log_file:
        log_file.write(f'[{current_time}] {message}\n')
    if console_output:
        print(f'[{current_time}] {message}')

def __create_log_file_if_inexistent():
    if not os.path.exists('log.txt'):
        with open('log.txt', 'w') as log_file:
            log_file.write('Log file created\n')