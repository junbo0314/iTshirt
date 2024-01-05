from itertools import product
import numpy as np
import pandas as pd



def irl(feature_matrix, n_actions, discount, transition_probability, 
        trajectories, epochs, learning_rate, stop) :
    """
    Find the reward function for the given trajectories.

    feature_matrix: Matrix with the nth row representing the nth state. NumPy
        array with shape (N, D) where N is the number of states and D is the
        dimensionality of the state.
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
        + 2 is state and action!!!
    epochs: Number of gradient descent steps. int.
    learning_rate: Gradient descent learning rate. float.
    -> Reward vector with shape (N,).
    """

    n_states, d_states = feature_matrix.shape

    #initialize weights
    np.random.seed(8)
    theta = np.random.uniform(size=(d_states,)) # number of dimension

    # Calculate the feature expectations \tilde{f}, from expert data
    feature_expectations = find_feature_expectations(feature_matrix, 
                                                     trajectories)
    
    # Gradient-based optimization on theta
    for i in range(epochs) :
        r = feature_matrix.dot(theta)   # (N,D) dot (D,1) -> (N,) vector
        expected_svf = find_expected_svf(n_states, n_actions, r, discount, stop,
                                         transition_probability, trajectories)

        gradient = feature_expectations - feature_matrix.T.dot(expected_svf)

        theta += learning_rate * gradient
    print(feature_expectations)
    print(expected_svf)
    policy, v = find_policy(n_states, n_actions, r, discount,
                        transition_probability=transition_probability)
    print(policy[32:], v[32:])

    

    return feature_matrix.dot(theta).reshape((n_states,))


def find_feature_expectations(feature_matrix, trajectories) :
    """
    Find the feature expectations for the given trajectories. This is the
    average path feature vector.

    feature_matrix: Matrix with the nth row representing the nth state. NumPy
        array with shape (N, D) where N is the number of states and D is the
        dimensionality of the state.
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> Feature expectations vector with shape (D,).
    """

    d_states = feature_matrix.shape[1]
    feature_expectations = np.zeros(d_states)

    for trajectory in trajectories :
        feature_expectations += feature_matrix[trajectory[0][0]]

        for _,_, next_state in trajectory :
            feature_expectations += feature_matrix[next_state]

    # Divide by the number of trajectories
    feature_expectations /= (len(trajectories))

    return feature_expectations



def find_expected_svf(n_states, n_actions, r, discount, stop, 
                      transition_probability, trajectories) :
    """
    Find the expected state visitation frequencies using algorithm 1 from
    Ziebart et al. 2008.

    n_states: Number of states N. int.
    r: Reward. NumPy array with shape (N,).
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> Expected state visitation frequencies vector with shape (N,).
    """

    n_trajectories = len(trajectories)
  
    # step 3
    policy, v = find_policy(n_states, n_actions, r, discount,
                        transition_probability=transition_probability)
    
    # step 4
    start_state_count = np.zeros(n_states)
    for trajectory in trajectories :
        start_state_count[trajectory[0][0]] += 1
    # divide by number of trajectories to make probability
    p_start_state = start_state_count / n_trajectories

    # transpose -> row: states, column : trajectories
    expected_svf = np.tile(p_start_state, (21, 1)).T

    # step 5
    for t in range(1, 21) :
        expected_svf[:, t] = 0
    
        for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
            if i != stop :
                expected_svf[k, t] += (expected_svf[i, t-1] * policy[i,j] * 
                                        transition_probability[i, j, k])
            
        expected_svf[:, t] /= np.sum(expected_svf[:,t])
  
            
    return expected_svf.sum(axis=1)


def optimal_value(n_states, n_actions, r, discount, threshold=1e-2,
                  transition_probability=None) :
    """
    Find the optimal value function. * using Bellman equation

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """

    v = np.zeros(n_states)

    diff = float("inf")

    while diff > threshold :
        diff = 0
        for state in range(n_states) :
            max_v = float("-inf")
            for action in range(n_actions) :
                tp = transition_probability[state,action,:]
                # Find max Q value of each state
                max_v = max(max_v, np.dot(tp, np.exp(r) + discount*v))

            new_diff = abs(v[state] - max_v)
            if new_diff > diff :
                diff = new_diff
            # Update the state value to max_v
            v[state] = max_v
    v /= np.min(v)
    return v



def find_policy(n_states, n_actions, r, discount, transition_probability,
                threshold=1e-2, v=None, stochastic=True) :
    """
    Find the optimal policy.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    v: Value function (if known). Default None.
    stochastic: Whether the policy should be stochastic. Default True.
    -> Action probabilities for each state or action int for each state
        (depending on stochasticity).
    """

    v = optimal_value(n_states, n_actions, r, discount, threshold=1e-2, 
                          transition_probability=transition_probability)
        
    if stochastic :

        '''
        Q = np.zeros((n_states,n_actions))
        for state in range(n_states) :
            for action in range(n_actions) :
                tp = transition_probability[state,action,:]
                Q[state,action] = tp.dot(np.exp(r+ discount*v))

        Q = Q/Q.sum(axis=1).reshape((n_states,1))

        '''

        diff = float("inf")

        while diff > threshold :
            diff = 0
            Q = np.zeros((n_states,n_actions))
            for state in range(n_states) :
                for action in range(n_actions) :
                    tp = transition_probability[state,action,:]
                    Q[state,action] = tp.dot(np.exp(r)+ discount*v)
            
            Q = Q/Q.sum(axis=1).reshape((n_states,1))
            
            for state in range(n_states) :
                value = 0
                for action in range(n_actions) :
                    tp = transition_probability[state,action,:]
                    value += Q[state,action]*tp.dot(np.exp(r)+ discount*v)
                new_diff = abs(v[state] - value)
                if new_diff > diff :
                    diff = new_diff
                v[state] = value
        v /= np.min(v)

            
        return Q, v
    

    
    
    
