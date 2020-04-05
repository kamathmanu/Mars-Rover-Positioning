import numpy as np
import graphics
import rover

'''
Utilities to save intermediate files
'''
def save_checkpoint(filename, filedata):
    import pickle

    print(f"saving: {filename}")
    with open(filename, "wb") as open_file:
        pickle.dump(filedata, open_file)

def load_checkpoint(filename):
    import pickle

    print(f"loading: {filename}")
    with open(filename, "rb") as open_file:
        return pickle.load(open_file)
'''
Utilities to return whether the distribution satisfies the axiom of probability
'''
def distribution_sum(prob_distribution):
    total_prob = 0
    for state in prob_distribution:
        total_prob = total_prob + prob_distribution[state]
    return total_prob

def distribution_sanity_check(prob_distribution):
    print("Sum of distribution: ", distribution_sum(prob_distribution))
##############################################################
def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations, missing):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)
    missing: A flag to determine which sanity check to print
    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """
    #print(len(all_possible_hidden_states))
    #print(observation_model(all_possible_hidden_states[17]))
    #print(transition_model(all_possible_hidden_states[17]))
    #print (observations)
    num_time_steps = len(observations)
    #print(num_time_steps)
    
    forward_messages = [None] * num_time_steps
    forward_messages[0] = rover.Distribution()
    for state in all_possible_hidden_states:
        P_obs = observation_model(state)
        '''
        if observations[0] == None:
            P_obs[[observations[0]] = 1
            P_obs.renormalize()
        '''
        forward_messages[0][state] = prior_distribution[state] * P_obs[observations[0]]
    forward_messages[0].renormalize()
    distribution_sanity_check(forward_messages[0])
    
    # Compute the forward messages
    print ("\nF/B: Computing forward messages:\n")
    for i in range(1, num_time_steps):        
        forward_messages[i] = rover.Distribution()
        for z_i in all_possible_hidden_states:
            innerSum = 0
            P_obs_given_hidden = (observation_model(z_i))[observations[i]]
            if (P_obs_given_hidden == 0):
                continue
            #print(P_obs_given_hidden)
            for z_iPrev in (forward_messages[i-1]):
                alpha_prev = (forward_messages[i-1])[z_iPrev]
                P_zn_given_zn_minus_one = transition_model(z_iPrev)[z_i]
                if (alpha_prev == 0 or P_zn_given_zn_minus_one == 0):
                    continue
                innerSum = innerSum + (alpha_prev * P_zn_given_zn_minus_one)                
            #print (innerSum)
            (forward_messages[i])[z_i] = P_obs_given_hidden * innerSum
        forward_messages[i].renormalize()
        #distribution_sanity_check(forward_messages[i])
    
    #save_checkpoint("FBalphas", forward_messages)
    
    # TODO: Compute the backward messages
    print ("\nF/B: Computing backward messages:\n")
    backward_messages = [None] * num_time_steps
    backward_messages[num_time_steps - 1] = rover.Distribution()
    for z_i in all_possible_hidden_states:
        backward_messages[num_time_steps - 1][z_i] = 1
    backward_messages[num_time_steps - 1].renormalize()
    #distribution_sanity_check(backward_messages[num_time_steps - 1])

    for i in range(num_time_steps-2, -1, -1):
        backward_messages[i] = rover.Distribution()
        for z_iNext in (backward_messages[i+1]):
            obs_prob = observation_model(z_iNext)[observations[i+1]]
            beta_next = backward_messages[i+1][z_iNext]
            if (obs_prob == 0 or beta_next == 0):
                continue
            for z_i in all_possible_hidden_states:
                intermediateSum = 0
                trans_prob = transition_model(z_i)[z_iNext]
                if (trans_prob == 0):
                    continue
                intermediateSum = intermediateSum + (beta_next * obs_prob * trans_prob)
                backward_messages[i][z_i] = intermediateSum
        backward_messages[i].renormalize()
        #distribution_sanity_check(backward_messages[i])
    
    #save_checkpoint("FBbetas", backward_messages)
    # TODO: Compute the marginals 
    marginals = [None] * num_time_steps
    
    print("\nF/B: Computing marginals: \n")
    for i in range(num_time_steps):
        marginals[i] = rover.Distribution()
        for z_i in all_possible_hidden_states:
            marginals[i][z_i] = forward_messages[i][z_i] * backward_messages[i][z_i]
        marginals[i].renormalize()
        #distribution_sanity_check(marginals[i])
    
    if (missing == True):
        print ("Part 2 Sanity Check: ")
        sanity_z1 = (6, 7, 'right')
        print("\nMarginal for hidden state ", sanity_z1, " at time i=30 is: ", marginals[30][sanity_z1])
    elif (missing == False):
        sanity_z1 = (6, 5, 'right')
        sanity_z2 = (6, 5, 'down')
        print ("Part 1(b) Sanity Check: ")
        print("\nMarginal for hidden state ", sanity_z1, " at time i=1 is: ", marginals[1][sanity_z1])
        print("\nMarginal for hidden state ", sanity_z2, " at time i=1 is: ", marginals[1][sanity_z2])
    
    #save_checkpoint("FBgammas", marginals)
                   
    return marginals

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: Write your code here
    
    return estimated_hidden_states


if __name__ == '__main__':
   
    enable_graphics = True
    
    missing_observations = False
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()
    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations, missing_observations)
    
    print('\n')
    timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    '''
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    '''
    print('\n')
    '''
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])
    '''
    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    estimated_states = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()
        
