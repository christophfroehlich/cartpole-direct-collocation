import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import pandas as pd


# Define the dynamics of the cartpole system
def dynamics(x, u):
    dx = np.zeros_like(x)
    theta = x[:,1]
    theta_p = x[:,3]
    fe = np.append(u, 0)
    s_p = x[:,2]
    dx[:,0] = x[:,2]
    dx[:,1] = x[:,3]
    dx[:,2] = 0.1e1 / (-np.cos(theta) ** 2 * lS ** 2 * mS ** 2 + lS ** 2 * mS ** 2 + lS ** 2 * mS * mW + 4 * IzzS * mS + 4 * IzzS * mW) * (np.sin(theta) * lS ** 3 * mS ** 2 * theta_p ** 2 + 2 * np.cos(theta) * np.sin(theta) * g * lS ** 2 * mS ** 2 + 4 * IzzS * np.sin(theta) * lS * mS * theta_p ** 2 - 2 * dR * lS ** 2 * mS * s_p + 2 * fe * lS ** 2 * mS - 8 * IzzS * dR * s_p + 8 * IzzS * fe) / 2

    dx[:,3] = -(np.cos(theta) * np.sin(theta) * lS * mS * theta_p ** 2 - 2 * np.cos(theta) * dR * s_p + 2 * np.sin(theta) * g * mS + 2 * np.sin(theta) * g * mW + 2 * np.cos(theta) * fe) * lS * mS / (-np.cos(theta) ** 2 * lS ** 2 * mS ** 2 + lS ** 2 * mS ** 2 + lS ** 2 * mS * mW + 4 * IzzS * mS + 4 * IzzS * mW)

    
    return dx

# Define the objective function to minimize control effort
def objective(u):
    return np.sum(u**2)

# Define the dynamics defects
def dynamics_defects_theta(decision_variables):
    u = decision_variables[:N]
    x = decision_variables[N:].reshape((N+1, states_dim))
    
    # Calculate the dynamics
    x_dot = dynamics(x, u)
    
    # Calculate the approximation of integral using trapezoidal quadrature
    integral = ((x_dot[:-1] + x_dot[1:])) / 2 * dt
    
    # Calculate the state defects
    defects = []
    for i in range(N):
        defects.append(x[i+1][1] - x[i][1] - integral[i][1])
    return np.array(defects)

def dynamics_defects_pos(decision_variables):
    u = decision_variables[:N]
    x = decision_variables[N:].reshape((N+1, states_dim))
    
    # Calculate the dynamics
    x_dot = dynamics(x, u)
    
    # Calculate the approximation of integral using trapezoidal quadrature
    integral = ((x_dot[:-1] + x_dot[1:])) / 2 * dt
    
    # Calculate the state defects
    defects = []
    for i in range(N):
        defects.append(x[i+1][0] - x[i][0] - integral[i][0])
    return np.array(defects)

def dynamics_defects_pos_dot(decision_variables):
    u = decision_variables[:N]
    x = decision_variables[N:].reshape((N+1, states_dim))
    
    # Calculate the dynamics
    x_dot = dynamics(x, u)
    
    # Calculate the approximation of integral using trapezoidal quadrature
    integral = ((x_dot[:-1] + x_dot[1:])) / 2 * dt
    
    # Calculate the state defects
    defects = []
    for i in range(N):
        defects.append(x[i+1][2] - x[i][2] - integral[i][2])
    return np.array(defects)

def dynamics_defects_theta_dot(decision_variables):
    u = decision_variables[:N]
    x = decision_variables[N:].reshape((N+1, states_dim))
    
    # Calculate the dynamics
    x_dot = dynamics(x, u)
    
    # Calculate the approximation of integral using trapezoidal quadrature
    integral = ((x_dot[:-1] + x_dot[1:])) / 2 * dt
    
    # Calculate the state defects
    defects = []
    for i in range(N):
        defects.append(x[i+1][3] - x[i][3] - integral[i][3])
    return np.array(defects)

# Define the direct collocation optimization problem
def optimization_problem(x0, xf, N):
    
    # Initial guess for control inputs
    u_init = np.zeros(N)
    u_init = u_init
    
    # Initial guess for states
    x_init = np.zeros((N+1, states_dim))
    x_init[:, 0] = np.linspace(x0[0], xf[0], N+1)
    x_init[:, 1] = np.linspace(x0[1], xf[1], N+1)
    # Concatenate control inputs and states into a single decision variable
    initial_guess = np.concatenate([u_init, x_init.flatten()])
    
    # Define the optimization problem
    def problem(decision_variables):
        u = decision_variables[:N]
       
        obj_value = objective(u)
        
        return obj_value
    
    # Define the bounds for the decision variables (system input)
    bounds = [(u_min, u_max)] * N 
    state_bounds = [(None, None)] * (states_dim * (N+1))

    for i in range(0,N+1):
        state_bounds[states_dim*i] = (l_b,u_b)

    bounds = bounds + state_bounds    

    #Enforcing Bound constraint on initial and final states
    bounds[N] =     (x0[0], x0[0])
    bounds[N+1] =   (x0[1], x0[1])
    bounds[N+2] =   (x0[2], x0[2])
    bounds[N+3] =   (x0[3], x0[3])
    bounds[5*N] =   (xf[0], xf[0])
    bounds[5*N+1] = (xf[1], xf[1])
    bounds[5*N+2] = (xf[2], xf[2])
    bounds[5*N+3] = (xf[3], xf[3])

    # Define the constraints
    constraints = [{'type': 'eq', 'fun': dynamics_defects_theta},
                   {'type': 'eq', 'fun': dynamics_defects_pos},
                   {'type': 'eq', 'fun': dynamics_defects_pos_dot},
                   {'type': 'eq', 'fun': dynamics_defects_theta_dot},]
    
    # Solve the optimization problem
    result = minimize(problem, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints, 
                      options={'maxiter': 1000, 'disp': True})
    
    return result

# Define the initial and final states 
case = 1
if case == 1:
    # swingup
    x0 = [0.0, 0.0, 0.0, 0.0]  # initial position and velocity
    xf = [0.0, np.pi, 0.0, 0.0]  # final position and velocity
    filename = 'swingup'
    T = 8 # end time
    N = 80
if case == 2:
    # swingdown
    x0 = [0.0, np.pi, 0.0, 0.0]  # initial position and velocity
    xf = [0.0, 0.0, 0.0, 0.0]  # final position and velocity
    filename = 'swingdown'
    T = 5 # end time
    N = 80
if case == 3:
    # sidestep, low, right
    x0 = [0.0, 0.0, 0.0, 0.0]  # initial position and velocity
    xf = [1.0, 0.0, 0.0, 0.0]  # final position and velocity
    filename = 'sidestep_low_right'
    T = 2 # end time
    N = 80
if case == 4:
    # sidestep, low, left
    x0 = [1.0, 0.0, 0.0, 0.0]  # initial position and velocity
    xf = [0.0, 0.0, 0.0, 0.0]  # final position and velocity
    filename = 'sidestep_low_left'
    T = 2 # end time
    N = 80
if case == 5:
    # sidestep, up, right
    x0 = [0.0, np.pi, 0.0, 0.0]  # initial position and velocity
    xf = [1.0, np.pi, 0.0, 0.0]  # final position and velocity
    filename = 'sidestep_up_right'
    T = 2 # end time
    N = 80
if case == 6:
    # sidestep, up, left
    x0 = [1.0, np.pi, 0.0, 0.0]  # initial position and velocity
    xf = [0.0, np.pi, 0.0, 0.0]  # final position and velocity
    filename = 'sidestep_up_left'
    T = 2 # end time
    N = 80

# Define the constants
lS = 1.0 # length of the pendulum
wS = 0.02 # width of the pendulum
mW = 10.0 # mass of the cart
mS = lS*wS*wS*2750 # mass of the pendulum
g = 9.81 # acceleration due to gravity
IzzS = 1/12*mS*(wS*wS+lS*lS) # moment of inertia of the pendulum about its center of mass
dR = 0.3 # viscous friction coefficient
print("Pendulum: m2 = {:.3f}, J = {:.3f}".format(mS, IzzS))
states_dim = 4

#bounds
u_b = 50
l_b = -50
u_min = -100
u_max = 100

# Define the number of time steps
t = np.linspace(0, T, N+1)  # time grid
dt = t[1] - t[0]  # time step

# Solve the optimization problem
result = optimization_problem(x0, xf, N)

if result is not None and result.success:
    # Extract the optimal control inputs and states
    u_opt = result.x[:N]

    x_opt = result.x[N:].reshape((N+1, states_dim))

    # Print the optimal control inputs and states
    print("Optimal control inputs:")
    print(u_opt)
    print("Optimal states:")
    print(x_opt)

    # Plotting control variables
    plt.figure(figsize=(8, 6))
    plt.plot(t[:-1], u_opt, 'bo-')
    plt.xlabel('Time')
    plt.ylabel('Control Input')
    plt.title('Optimal Control Inputs')
    plt.grid(True)
    plt.show()

    # Plotting state variables
    t_ = np.linspace(0, T, N+1)  # time grid

    plt.figure(figsize=(8, 6))
    plt.plot(t_, x_opt[:, 0], label='Position')
    plt.plot(t_, x_opt[:, 1], label='Theta')
    plt.plot(t_, x_opt[:, 2], label='Velocity')
    plt.plot(t_, x_opt[:, 3], label='Theta_dot')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.title('Optimal State Variables')
    plt.legend()
    plt.grid(True)
    plt.show()

    cs = CubicSpline(t_, x_opt[:, 0])


    # Export the DataFrame to a CSV file
    df_x = pd.DataFrame(x_opt, columns=['Position', 'Theta', 'Velocity', 'Theta_dot'])
    # Create a new DataFrame from u_opt
    df_u = pd.DataFrame(u_opt, columns=['u'])
    # Create a new DataFrame from t
    df_t = pd.DataFrame(t_, columns=['t'])
    # Concatenate the new DataFrame with the existing one
    df = pd.concat([df_t, df_x, df_u], axis=1)
    # Export the DataFrame to a CSV file
    df.to_csv('{}.csv'.format(filename), index=False)
else:
    print("Optimization failed: {}", result.message)