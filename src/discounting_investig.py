import json
import numpy as np
from typing import List, Tuple, Union
from riverswim import RiverSwim
from submission import policy_iteration, value_iteration, bellman_backup

np.set_printoptions(precision=3)

def gen_discounting_space(
    policy_func: Union[policy_iteration, value_iteration],
    discount_factors: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    tol: float=1e-3
) -> List[dict]:
    
    data = [{}] * np.shape(discount_factors)[0]

    for i, discount_factor in enumerate(discount_factors):

        V, policy = policy_func(R, T, gamma=discount_factor, tol=tol)
        data[i] = dict(
            discount_factor=discount_factor,
            V=V,
            policy=policy
        )

    return data


def scan_dicts(
    discount_dicts: List[dict],
    state_index: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    num_dicts = len(discount_dicts)
    discount_values = np.zeros(num_dicts)
    action_at_state = np.nan * np.ones(num_dicts)
    expected_return_at_state = np.zeros(num_dicts)

    for i, discount_dict in enumerate(discount_dicts):
        discount_values[i] = discount_dict["discount_factor"]
        action_at_state[i] = discount_dict["policy"][state_index]
        expected_return_at_state[i] = discount_dict["V"][state_index]

    return discount_values, action_at_state, expected_return_at_state

def action_value_function(s, a, R, T, gamma, V):
    return np.inner(T[s, a, :], bellman_backup(s, a, R, T, gamma, V))

def locate_gamma(
    decimal_place: int,
    state_index: int,
    action_index: int,
    policy_func: Union[policy_iteration, value_iteration],
    R: np.ndarray,
    T: np.ndarray,
    tol: float 
) -> Tuple[np.ndarray, np.ndarray]:

    step_scale = 0.1
    step_sizes = np.array([step_scale ** k for k in range(1, (decimal_place + 1))])
    start_bound, stop_bound = 0, 1

    for i in range(0, decimal_place):

        discount_factors = np.arange(start_bound, stop_bound, step_sizes[i])
        discount_factors = np.flip(discount_factors, axis=0)

        data = gen_discounting_space(policy_func, discount_factors, R, T, tol)
        # print(f"Value of data_pi: {data_pi}")
        _, action_at_state_pi, expected_return_at_state_pi = scan_dicts(data, state_index=state_index)
        format_string = '.' + str(i + 1) + 'f'
        select_zeros = (action_at_state_pi == action_index)
        print(f"**** Results of step (inc={step_sizes[i]:{format_string}}) run over [{start_bound:{format_string}}, {stop_bound:{format_string}}) ****")
        print(f'Discounts associated with action {action_index} at state {state_index} policy: {discount_factors[select_zeros]}')
        # print(f'Expected rewards corresponding with "Left at leftmost state" policy: {expected_return_at_state_pi[select_zeros]}')
        
        dict_select = np.arange(0, select_zeros.shape[0])[select_zeros]
        associated_dict = data[dict_select[0]]
        print(f'Expected returns {associated_dict["V"]} for discount factor {associated_dict["discount_factor"]}')
        q_left = bellman_backup(state_index, action_index, R, T, associated_dict["discount_factor"], associated_dict["V"])
        q_right = bellman_backup(state_index, (action_index + 1), R, T, associated_dict["discount_factor"], associated_dict["V"])
        print(f"Expected return for action {action_index} at state {state_index}: {q_left}")
        print(f"Expected return for action {action_index + 1} at state {state_index}: {q_right}")

        next_associated_dict = data[dict_select[1]]    
        print(f'Expected returns {next_associated_dict["V"]} for discount factor {next_associated_dict["discount_factor"]}')
        
        start_bound = discount_factors[select_zeros][0]
        stop_bound = start_bound + step_sizes[i]

        """
        discount_factors = np.arange(start_bound, stop_bound, fine_step)
        discount_factors = np.flip(discount_factors, axis=0)

        data_pi = gen_discounting_space(policy_iteration, discount_factors, R, T)
        # print(f"Value of data_pi: {data_pi}")
        _, action_at_state_pi, expected_reward_at_state_pi = scan_dicts(data_pi, state_index=0)

        select_zeros = (action_at_state_pi == 0)
        print(f'**** Results of Fine step (inc={fine_step}) run over [{start_bound}, {stop_bound})')
        print(f'Discounts associated with "Left at state 1" policy: {discount_factors[select_zeros]}')
        print(f'Expected rewards corresponding with "Left at state 1" policy: {expected_reward_at_state_pi[select_zeros]}')
        """

    return discount_factors[select_zeros], expected_return_at_state_pi[select_zeros]


if __name__ == "__main__":
    SEED = 1234

    RIVER_CURRENT = 'WEAK'
    # RIVER_CURRENT = 'MEDIUM'
    # RIVER_CURRENT = 'STRONG'
    assert RIVER_CURRENT in ['WEAK', 'MEDIUM', 'STRONG']
    env = RiverSwim(RIVER_CURRENT, SEED)

    R, T = env.get_model()
    tol = 1e-3

    decimal_place = 3
    state_index = 0
    action_index = 0

    heading_pi = ('*' * 25) + ' Policy Iteration ' + ('*' * 25) 
    print(heading_pi)
    gammas_pi, expected_returns_pi = locate_gamma(
        decimal_place,
        state_index,
        action_index,
        policy_iteration,
        R,
        T,
        tol
    )

    heading_vi = ('*' * 25) + ' Value Iteration ' + ('*' * 25)
    print(heading_vi)
    gammas_vi, expected_returns_vi = locate_gamma(
        decimal_place,
        state_index,
        action_index,
        value_iteration,
        R,
        T,
        tol
    )