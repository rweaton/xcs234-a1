import numpy as np
from typing import Union
from riverswim import RiverSwim

np.set_printoptions(precision=3)

def bellman_backup(
    state: int,
    action: int,
    R: np.ndarray,
    T: np.ndarray,
    gamma: float,
    V: np.ndarray[:] 
) -> float:
    """
    Perform a single Bellman backup.

    Parameters
    ----------
    state: int
    action: int
    R: np.array Union[(num_states, num_actions), (num_states, num_actions, num_states)]
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    V: np.array (num_states)

    Returns
    -------
    backup_val: float
    """
    backup_val = None
    ############################
    ### START CODE HERE ###

    if R.axis == 2:
        # backup_val = R[state, action] + gamma * np.dot(T[state, action, :], V)
        backup_val = (
            R[state, action] + 
            gamma * np.inner(T[state, action, :], V)
        )

    elif R.axis == 3:
        backup_val = np.inner(
            (T[state, action, :]),
            (R[state, action, :] + gamma * V)
        )

    ### END CODE HERE ###
    ############################

    return backup_val


def value_iteration(
    R: np.ndarray,
    T: np.ndarray,
    gamma: float,
    terminator: Union[float, int]=1e-3
) -> np.ndarray:
    """Runs value iteration.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    value_function: np.array (num_states)
    policy: np.array (num_states)
    """

    if isinstance(terminator, float):
        termination_type = 'indefinite_to_tol'

    elif isinstance(terminator, int):
        termination_type = 'finite_horizon'

    num_states, num_actions = R.shape
    value_function = None
    policy = None

    # heading_vi = "Running value_iteration()..."
    # print(heading_vi)

    # initialize iteration sets
    states = np.arange(0, num_states)
    actions = np.arange(0, num_actions)

    # initialize targets for updating
    # policy = np.zeros(num_states)
    value_function = np.zeros(num_states)
    iteration_count = 1

    # force first iteration
    # norm_gt_tol = True
    do_iterate = True

    # while norm_gt_tol:
    while do_iterate:

        # report progress
        """
        if (iteration_count % 10) == 0:
            print(f"Beginning iteration: {iteration_count}")
            current_policy = [['L', 'R'][a] for a in policy]
            print(f"Current policy: {current_policy}")
        """
        
        single_bell_backups = np.array(
            [
                [
                    bellman_backup(s, a, R, T, gamma, value_function)
                    for a in actions
                ] 
                for s in states
            ]
        )
        # print(f"Shape of single_bell_backups: {single_bell_backups.shape}")
        policy = np.argmax(single_bell_backups, axis=1)
        # print(f"Shape of policy: {policy.shape}")
        value_function_next = np.array(
            [single_bell_backups[s, policy[s]] for s in states]
        )

        if termination_type == 'indefinite_to_tol':
            # print(f"Shape of value_function_next: {value_function_next.shape}")
            do_iterate = np.linalg.norm(
                value_function_next  - value_function,
                ord=np.inf
            ) > terminator

        elif termination_type == 'finite_horizon':
            do_iterate = (iteration_count <= terminator)

        if do_iterate == False:
            
            # iterate a final step using V_next
            single_bell_backups = np.array(
                [
                    [
                        bellman_backup(s, a, R, T, gamma, value_function_next)
                        for a in actions
                    ] 
                    for s in states
                ]
            )
            policy = np.argmax(single_bell_backups, axis=1)          
            value_function = value_function_next
            
        else:
            value_function = value_function_next
            iteration_count += 1

    return value_function, policy