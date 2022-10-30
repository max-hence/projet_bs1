import numpy as np
import random

def fhn_modelling(u_init:float, v_init:float, B:float, dt:float=1, time:int=400):
    """ Reproduces Figure 2: Modelization of the bisatabilty of the model FHN

    Args:
        u_init (float): u value to init
        v_init (float): v value to init
        B (float): bistability degree
        dt (float, optional): time step. Defaults to 0.1.
        time (int, optional): total time of the modelization. Defaults to 400.

    Returns:
        numpy.ndarray object, representing the evolution of u and v curves
    """
    # Parameters
    A = 0.1 #
    E = 0.01 #
    N = int(time / dt)

    # Initialization
    mat_U, mat_V = np.zeros(N), np.zeros(N)
    mat_U[0], mat_V[0] = u_init, v_init

    # Algorithm
    for i in range(0, N-1):
        mat_U[i+1] = mat_U[i] + (mat_U[i] - mat_U[i]**3 - mat_V[i])*dt
        mat_V[i+1] = mat_V[i] + (E*(mat_U[i] - B*mat_V[i] + A))*dt

    return mat_U, mat_V

def u_nullcline(u):
    return u - u**3

def v_nullcline(v, B):
    return B*v - 0.1

def fhn_space(u_init:float=-0.6, v_init:float=-0.3, B:float=1, dt:float=1, time:int=400, width:int=400):
    # Parameters
    A = 0.1
    E = 0.01

    #Initialization
    #Each position has a heat profile in time
    mat_U, mat_V = np.full((width, time), u_init), np.full((width, time), v_init)
    mat_U[180:220, 0] = 1

    #Algorithm
    for j in range(0, time-1): #for each timepoint
        for i in range(1, width-1): #for each position
            mat_U[i, j + 1] = mat_U[i, j] + (mat_U[i, j] - mat_U[i, j]**3 - mat_V[i, j])*dt
            mat_V[i, j + 1] = mat_V[i, j] + (E * (mat_U[i, j] - B*mat_V[i, j] + A))*dt
    return mat_U

def fhn_space_diffusion(u_init:float=-0.6, v_init:float=-0.3, B:float=1, D:float=1, dt:float=1, dx:float=1, time:int=400, width:int=400):
    # Parameters
    A = 0.1
    E = 0.01

    #INIT
    #each position has a heat profile in time
    mat_U, mat_V = np.full((width, time), u_init), np.full((width, time), v_init)
    mat_U[180:220, :] = 1

    #ALGO
    for j in range(0, time-1): #for each timepoint
        for i in range(1, width-1): #for each position
            mat_U[i, j + 1] = mat_U[i, j] + (D*dt/dx**2) * (mat_U[i+1, j] - 2*mat_U[i,j] + mat_U[i-1, j]) + (mat_U[i, j] - mat_U[i, j]**3 - mat_V[i, j]) * dt
            mat_V[i, j + 1] = mat_V[i, j] + (D*dt/dx**2) * (mat_V[i+1, j] - 2*mat_V[i, j] + mat_V[i-1, j]) + (E * (mat_U[i, j] - B*mat_V[i, j] + A)) * dt
    return mat_U
