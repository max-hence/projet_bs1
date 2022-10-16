import numpy as np

def fhn_modelling(u_init:float, v_init:float, B:float, dt:float=0.1, time:int=400):
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
    mat_U = np.zeros(N)
    mat_V = np.zeros(N)
    mat_U[0] = u_init
    mat_V[0] = v_init

    # Algorithm
    for i in range(0, N-1):
        mat_U[i+1] = mat_U[i] + (mat_U[i] - mat_U[i]**3 - mat_V[i])*dt
        mat_V[i+1] = mat_V[i] + (E*(mat_U[i] - B*mat_V[i] + A))*dt

    return mat_U, mat_V