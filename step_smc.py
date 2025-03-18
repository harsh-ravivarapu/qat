import numpy as np
from scipy.special import softmax
import scipy.io
import pickle

from utils import create_SMC

def step_function_SMC_step(Action, kk, IT, freq, len, b, dt, sliding):
    # Load the CTX_workspace saved variable
    with open('CTX_workspace.pkl', 'rb') as f:
        CTX_workspace = pickle.load(f)

    # Action logits to Action
    Action = np.reshape(Action, (len, b))
    Action = softmax(Action, axis=0)
    idx = np.argmax(Action, axis=0)
    pprofile = np.zeros(len)
    pprofile[idx] = 1
    tmax = len

    # Simulation (for create_SMC and rl_BGM_step_SMC_pulse_python_step, you'll need to have these functions translated to Python)
    SMC_pulse = create_SMC(14, (tmax / 0.01) + 1, dt, 5, 0, 3.5)
    beta_vec, EI, kk, CTX_workspace = rl_BGM_step_SMC_pulse_python_step(
        tmax, IT, 1, 0, freq, pprofile, kk, len, SMC_pulse, CTX_workspace, dt, sliding)

    # Load info
    beta_vec /= 650

    # Verify lengths
    ei_len = len(EI)
    beta_len = len(beta_vec)

    if ei_len != beta_len:
        scipy.io.savemat('last_workspace.mat', mdict={'CTX_workspace': CTX_workspace})

    LoggedSignal_State = np.concatenate([beta_vec, EI])  # This replaces LoggedSignal.State
    IT += 1
    Observation = LoggedSignal_State

    # Reward Function
    isdone = False
    if np.mean(beta_vec) > 650 / 650:
        isdone = True
        Reward = -100
    elif np.mean(beta_vec) > 500 / 650:
        Reward = -10
    elif np.mean(beta_vec) > 400 / 650:
        Reward = -1
    elif np.mean(beta_vec) > 350 / 650:
        Reward = 1
    elif np.mean(beta_vec) > 325 / 650:
        Reward = 10
    else:
        Reward = 100

    return Observation, Reward, isdone, kk, IT
