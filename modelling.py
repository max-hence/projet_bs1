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
    #¬†Parameters
    A = 0.1 #
    E = 0.01 #
    N = int(time / dt)

    #¬†Initialization
    mat_U, mat_V = np.zeros(N), np.zeros(N)
    mat_U[0], mat_V[0] = u_init, v_init

    #¬†Algorithm
    for i in range(0, N-1):
        mat_U[i+1] = mat_U[i] + (mat_U[i] - mat_U[i]**3 - mat_V[i])*dt
        mat_V[i+1] = mat_V[i] + (E*(mat_U[i] - B*mat_V[i] + A))*dt

    return mat_U, mat_V

def u_nullcline(u):
    return u - u**3

def v_nullcline(v, B):
    return B*v - 0.1

def fhn_space_diffusion(u_init:float=-0.6, v_init:float=-0.3, start:tuple=(1800, 2200),
                        B:float=1, D:float=1, dt:float=1, dx:float=1, time:int=400, width:int=400):
    # Parameters
    A = 0.1
    E = 0.01
    F = (D*dt/dx**2)
    #INIT
    #each position has a heat profile in time
    mat_U, mat_V = np.full((width, time), u_init), np.full((width, time), v_init)
    mat_U[start[0]:start[1]] = 1

    #ALGO
    for j in range(0, time-1):
        mat_U[0, j + 1] = mat_U[0, j] + F * (mat_U[0+1, j] - mat_U[0, j])  +   (mat_U[0, j] - mat_U[0, j]**3 - mat_V[0, j])*dt
        mat_V[0, j + 1] = mat_V[0, j] + F * (mat_V[0+1, j] - mat_V[0, j])   +   (E * (mat_U[0, j] - B*mat_V[0, j] + A))*dt

        for i in range(1, width-1):
            mat_U[i, j + 1] = mat_U[i, j] + F * (mat_U[i+1, j] - 2*mat_U[i, j] + mat_U[i-1, j])  + (mat_U[i, j] - mat_U[i, j]**3 - mat_V[i, j])*dt
            mat_V[i, j + 1] = mat_V[i, j] + F * (mat_V[i+1, j] - 2*mat_V[i, j] + mat_V[i-1, j])  + (E * (mat_U[i, j] - B*mat_V[i, j] + A))*dt

        mat_U[width-1, j + 1] = mat_U[width-1, j] + F * (mat_U[width-1-1, j] - mat_U[width-1, j])  +   (mat_U[width-1, j] - mat_U[width-1, j]**3 - mat_V[width-1, j])*dt
        mat_V[width-1, j + 1] = mat_V[width-1, j] + F * (mat_V[width-1-1, j] - mat_V[width-1, j])  +   (E * (mat_U[width-1, j] - B*mat_V[width-1, j] + A))*dt
    return mat_U

def self_organized(width:int = 2000, time=10000):
    # Parameters
    A = 0.1
    E = 0.01
    F = (1*0.1/1**2)
    dt = 0.1
    B = 1
    u_init = -0.6
    v_init = -0.3

    #INIT
    #each position has a heat profile in time
    mat_U, mat_V = np.full((width, time), u_init), np.full((width, time), v_init)
    for i in range(width):
        mat_U[i, 0] = random.uniform(-0.6, -0.2)

    #ALGO
    for j in range(0, time-1):
        mat_U[0, j + 1] = mat_U[0, j] + F * (mat_U[0+1, j] - mat_U[0, j])  +   (mat_U[0, j] - mat_U[0, j]**3 - mat_V[0, j])*dt
        mat_V[0, j + 1] = mat_V[0, j] + F * (mat_V[0+1, j] - mat_V[0, j])   +   (E * (mat_U[0, j] - B*mat_V[0, j] + A))*dt

        for i in range(1, width-1):
            mat_U[i, j + 1] = mat_U[i, j] + F * (mat_U[i+1, j] - 2*mat_U[i, j] + mat_U[i-1, j])  + (mat_U[i, j] - mat_U[i, j]**3 - mat_V[i, j])*dt
            mat_V[i, j + 1] = mat_V[i, j] + F * (mat_V[i+1, j] - 2*mat_V[i, j] + mat_V[i-1, j])  + (E * (mat_U[i, j] - B*mat_V[i, j] + A))*dt

        mat_U[width-1, j + 1] = mat_U[width-1, j] + F * (mat_U[width-1-1, j] - mat_U[width-1, j])  +   (mat_U[width-1, j] - mat_U[width-1, j]**3 - mat_V[width-1, j])*dt
        mat_V[width-1, j + 1] = mat_V[width-1, j] + F * (mat_V[width-1-1, j] - mat_V[width-1, j])  +   (E * (mat_U[width-1, j] - B*mat_V[width-1, j] + A))*dt

    return mat_U