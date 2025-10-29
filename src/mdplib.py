import numpy as np
from typing import Union
from inventory import Inventory

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

    if len(R.shape) == 2:
        # backup_val = R[state, action] + gamma * np.dot(T[state, action, :], V)
        backup_val = (
            R[state, action] + 
            gamma * np.inner(T[state, action, :], V)
        )

    elif len(R.shape) == 3:
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
        # record = None

    elif isinstance(terminator, int):
        termination_type = 'finite_horizon'
        # record = [{} for _ in range(0, terminator + 1)]
    # print(f"Value of termination_type: {termination_type}")

    if len(R.shape) == 2:
        num_states, num_actions = R.shape
    elif len(R.shape) == 3:
        num_states, num_actions, _ = R.shape

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
        # if termination_type == 'finite_horizon':
        #     record[iteration_count - 1] = dict(
        #         current_value_function=value_function,
        #         policy=policy,
        #     )

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
            do_iterate = (iteration_count < terminator)

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

            # if termination_type == 'finite_horizon':
            #     record[iteration_count] = dict(
            #         current_value_function=value_function,
            #         policy=policy,
            #     )

        else:
            value_function = value_function_next
            iteration_count += 1

    # return value_function, policy, record
    return value_function, policy

if __name__ == "__main__":
    SEED = 1234
    INITIAL_STATE = 3

    env = Inventory(INITIAL_STATE, SEED)

    R, T = env.get_model()
    # discount_factor = 0.99
    # discount_factor = 1
    # terminator = 11
    # discount_factor = 0.9
    discount_factor = 0.51
    terminator = 1e-3
    print(f"Value of R: {R}")
    print(f"Value of T: {T}")
    state = 2
    action = 1
    # V_test = (-5.) * np.ones(T.shape[0])
    # backup_value = bellman_backup(state, action, R, T, discount_factor, V_test)
    # print(f"Bellman Backup value for state {state}, action {action} is {backup_value}.")

    # value_function, policy = value_iteration(R, T, discount_factor, tol=1e-3)
    # print(f"Value function after value_iteration: {value_function}")
    # print(f"Policy after value_iteration: {policy}")
    if isinstance(terminator, int):
        terminators = reversed(range(1, terminator + 1))
        print(f"Value of terminators: {terminators}")
        print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)
        print(f"Using discount factor: {discount_factor}")
        for terminator in terminators:
            print(f"\nUsing terminator value: {terminator}")
            V_vi, policy_vi = value_iteration(R, T, gamma=discount_factor, terminator=terminator)
            policy_assignments = [(i, ['sell', 'buy'][a]) for i, a in enumerate(policy_vi)]
            print(f"State and action for that state for policy (s, pi[a]):\n {policy_assignments}")
            expected_returns = [(i, f"{v:.3f}") for i, v in enumerate(V_vi)]
            print(f"State and its associated expected return (s, V_vi[s]):\n {expected_returns}")

    else:
        V_vi, policy_vi = value_iteration(R, T, gamma=discount_factor, terminator=terminator)
        print(f"\nUsing terminator value: {terminator}")
        V_vi, policy_vi = value_iteration(R, T, gamma=discount_factor, terminator=terminator)
        policy_assignments = [(i, ['sell', 'buy'][a]) for i, a in enumerate(policy_vi)]
        print(f"State and action for that state for policy (s, pi[a]):\n {policy_assignments}")
        expected_returns = [(i, f"{v:.3f}") for i, v in enumerate(V_vi)]
        print(f"State and its associated expected return (s, V_vi[s]):\n {expected_returns}")

    """
    if record:
        for iteration, vi_pair in enumerate(record):
            V_vi = vi_pair["current_value_function"]
            policy_vi = vi_pair["policy"]
            print(f"Iteration: {iteration}")
            expected_returns = [(i, f"{v:.3f}") for i, v in enumerate(V_vi)]
            print(f"State and its associated expected return (s, V_vi[s]): {expected_returns}")
            policy_assignments = [(i, ['sell', 'buy'][a]) for i, a in enumerate(policy_vi)]
            print(f"State and action for that state for policy (s, pi[a]): {policy_assignments}")
    else:
        expected_returns = [(i, f"{v:.3f}") for i, v in enumerate(V_vi)]
        print(f"State and its associated expected return (s, V_vi[s]): {expected_returns}")
        policy_assignments = [(i, ['sell', 'buy'][a]) for i, a in enumerate(policy_vi)]
        print(f"State and action for that state for policy (s, pi[a]): {policy_assignments}")
    """