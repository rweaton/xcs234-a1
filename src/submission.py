### MDP Value Iteration and Policy Iteration

import numpy as np
from riverswim import RiverSwim

np.set_printoptions(precision=3)

def bellman_backup(state, action, R, T, gamma, V):
    """
    Perform a single Bellman backup.

    Parameters
    ----------
    state: int
    action: int
    R: np.array (num_states, num_actions)
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

    backup_val = R[state, action] + gamma * np.dot(T[state, action, :], V)

    ### END CODE HERE ###
    ############################

    return backup_val

def policy_evaluation(policy, R, T, gamma, tol=1e-3):
    """
    Compute the value function induced by a given policy for the input MDP
    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    tol: float

    Returns
    -------
    value_function: np.array (num_states)
    """
    num_states, _ = R.shape
    value_function = None

    ############################
    ### START CODE HERE ###

    # initialize iteration index
    k = 1

    # generate set of states
    states = np.arange(0, num_states)

    # set initial expected return functions to zero 
    # for each state
    value_function = np.zeros(num_states)
    new_value_function = np.zeros(num_states)

    # force first iteration
    norm_gt_tol = True

    # Iteration block
    while norm_gt_tol:

        # obtain single Bellman backups for all 
        # possible state-action pairs
        for state in states:
            new_value_function[state] = bellman_backup(
                state, policy[state], R, T, gamma, value_function
            )

        # Determine if new_value_function has diverged 
        # from current value_function above tolerance 
        norm_gt_tol = np.linalg.norm(
            new_value_function - value_function,
            ord=2
        ) > tol

        # If sufficient divergence, set up for next iteration
        if norm_gt_tol:

            # set new value function to current status 
            # for next iteration
            value_function = new_value_function

            # update iteration index for next iteration
            k += 1

        ### NOTE: make sure value function and policy correctly correspond to each other!!!


    ### END CODE HERE ###
    ############################
    return value_function


def policy_improvement(R, T, V_policy, gamma):
    """
    Given the value function induced by a given policy, perform policy improvement
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    V_policy: np.array (num_states)
    gamma: float

    Returns
    -------
    new_policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    new_policy = None

    ############################
    ### START CODE HERE ###

    # Initialize state_action value Q_pi, states 
    # actions sets for iteration block
    Q_pi = np.zeros((num_states, num_actions))
    states = np.arange(0, num_states)
    actions = np.arange(0, num_actions)
    
    # Compute expected return for each possible
    # state-action pair under current policy
    for state in states:
        for action in actions:
            Q_pi[state, action] = bellman_backup(
                state, action, R, T, gamma, V_policy
            )

    # Produce new policy by selecting the action with 
    # the most expected reward for each state
    new_policy = np.argmax(Q_pi, axis=1)
    ### END CODE HERE ###
    ############################
    return new_policy


def policy_iteration(R, T, gamma, tol=1e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    V_policy: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, _ = R.shape
    V_policy = None
    policy = None
    ############################
    ### START CODE HERE ###

    # set iteration index
    i = 0

    # Randomly generate initial policy
    _, num_actions = R.shape
    actions_set = np.arange(0, num_actions)
    policy = np.random.choice(
        actions_set,
        size=num_states,
        replace=True
    )

    # force first iteration
    norm_gt_tol = True

    # iteration block
    while norm_gt_tol:

        # Calculate expected return for each state of current policy
        V_policy = policy_evaluation(policy, R, T, gamma, tol)

        # Use expected returns to update current policy to new policy
        new_policy = policy_improvement(R, T, V_policy, gamma)

        # Determine if new_policy has deviated from current policy above tolerance
        norm_gt_tol = np.linalg.norm(
            new_policy - policy,
            ord=1
        ) > tol

        if norm_gt_tol:
            # set new policy to current policy for next iteration
            policy = new_policy
            # update iteration index for next iteration
            i += 1

        #### NOTE: make sure policy and V_policy correctly correspond!!!

    ### END CODE HERE ###
    ############################
    return V_policy, policy


def value_iteration(R, T, gamma, tol=1e-3):
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
    num_states, num_actions = R.shape
    value_function = None
    policy = None
    ############################
    ### START CODE HERE ###

    # initialize iteration sets
    states = np.arange(0, num_states)
    actions = np.arange(0, num_actions)

    # initialize targets for updating
    # policy = np.zeros(num_states)
    value_function = np.zeros(num_states)
    iteration_count = 1

    # force first iteration
    norm_gt_tol = True

    while norm_gt_tol:

        # report progress
        if (iteration_count % 10) == 0:
            print(f"Beginning iteration: {iteration_count}")
            current_policy = [['L', 'R'][a] for a in policy]
            print(f"Current policy: {current_policy}")
        
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

        # print(f"Shape of value_function_next: {value_function_next.shape}")
        norm_gt_tol = np.linalg.norm(
            value_function_next  - value_function,
            ord=np.inf
        ) > tol


        if norm_gt_tol == False:
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

        else:
            value_function = value_function_next
            iteration_count += 1

    ### END CODE HERE ###
    ############################
    return value_function, policy


# Edit below to run policy and value iteration on different configurations
# You may change the parameters in the functions below
if __name__ == "__main__":
    SEED = 1234

    RIVER_CURRENT = 'WEAK'
    # RIVER_CURRENT = 'MEDIUM'
    # RIVER_CURRENT = 'STRONG'
    assert RIVER_CURRENT in ['WEAK', 'MEDIUM', 'STRONG']
    env = RiverSwim(RIVER_CURRENT, SEED)

    R, T = env.get_model()
    discount_factor = 0.99
    # print(f"Value of R: {R}")
    # print(f"Value of T: {T}")
    state = 2
    action = 0
    V_test = (-5.) * np.ones(T.shape[0])
    backup_value = bellman_backup(state, action, R, T, discount_factor, V_test)
    print(f"Bellman Backup value for state {state}, action {action} is {backup_value}.")

    # value_function, policy = value_iteration(R, T, discount_factor, tol=1e-3)
    # print(f"Value function after value_iteration: {value_function}")
    # print(f"Policy after value_iteration: {policy}")
    
    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)

    V_pi, policy_pi = policy_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_pi)
    print([['L', 'R'][a] for a in policy_pi])
    """
    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)

    V_vi, policy_vi = value_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_vi)
    print([['L', 'R'][a] for a in policy_vi])
    """