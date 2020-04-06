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
###############################################################################

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
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

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps 
    
    #Initialize forward(z0)
    forward_messages[0] = rover.Distribution()
    for z_i in all_possible_hidden_states:
        obs_distr = observation_model(z_i)
        observation = observations[0]
        P_obs = 1
        if observation is not None:
            P_obs = obs_distr.get(observation)
        if P_obs is None:
            continue
        forward_messages[0][z_i] = prior_distribution[z_i] * P_obs
    forward_messages[0].renormalize()
    #distribution_sanity_check(forward_messages[0])
    
    #Compute the forward messages
    print ("\nF/B: Computing forward messages:\n")
    for i in range(1, num_time_steps):
        forward_messages[i] = rover.Distribution()
        for z_i in all_possible_hidden_states:
            innerSum = 0
            #The observation probability distribution
            obs_distr = observation_model(z_i)
            #The observation itself
            observation = observations[i]
            #The probability of observation, given a hidden state.
            #Assume the observation is missing, assigning it unit probability
            P_obs_given_hidden = 1
            #if the observation exists, get it from the distribution model
            if observation is not None:
                P_obs_given_hidden = obs_distr.get(observation)
            #If the observation didn't exist, the inferred hidden state wouldn't 
            #be possible based on this and as such we ignore it
            if (P_obs_given_hidden is None):
                continue
            for z_iPrev in (forward_messages[i-1]):
                alpha_prev = (forward_messages[i-1])[z_iPrev]
                P_zn_given_zn_minus_one = transition_model(z_iPrev)[z_i]
                #The branch below shouldn't really ever execute? IDK
                if (P_zn_given_zn_minus_one == 0):
                    continue
                innerSum = innerSum + (alpha_prev * P_zn_given_zn_minus_one)
            forward_messages[i][z_i] = P_obs_given_hidden * innerSum
        forward_messages[i].renormalize()
        #distribution_sanity_check(forward_messages[i])
    
    print ("\nF/B: Computing backward messages:\n")
    #Initialize the beta(zn-1) to a uniform distribution
    backward_messages[num_time_steps - 1] = rover.Distribution()
    for z_i in all_possible_hidden_states:
        backward_messages[num_time_steps - 1][z_i] = 1
    backward_messages[num_time_steps - 1].renormalize()
    #distribution_sanity_check(backward_messages[num_time_steps - 1])
    
    #Compute the backward messages
    for i in range(num_time_steps-2, -1, -1):
        backward_messages[i] = rover.Distribution()
        for z_iNext in (backward_messages[i+1]):
            #The observation probability distribution
            obs_distr = observation_model(z_iNext)
            #The observation itself
            observation = observations[i+1]
            #The probability of observation, given a hidden state. 
            #Assume the observation is missing, assigning it unit probability
            obs_prob = 1
            #if the observation exists, get it from the distribution model
            if observation is not None:
                obs_prob = obs_distr.get(observation)
            #If the observation didn't exist, the inferred hidden state wouldn't
            #be possible based on this and as such we ignore it
            if (obs_prob is None):
                continue
            prod = backward_messages[i+1][z_iNext] * obs_prob
            for z_i in all_possible_hidden_states:
                trans_prob = transition_model(z_i)[z_iNext]
                if (trans_prob == 0):
                    continue
                backward_messages[i][z_i] = (prod * trans_prob)
        backward_messages[i].renormalize()
        #distribution_sanity_check(backward_messages[i])
                
    #Compute the marginals 
    print("\nF/B: Computing marginals: \n")
    for i in range(num_time_steps):
        marginals[i] = rover.Distribution()
        for z_i in all_possible_hidden_states:
            marginals[i][z_i] = forward_messages[i][z_i] * backward_messages[i][z_i]
        marginals[i].renormalize()
        #distribution_sanity_check(marginals[i])
    '''
    sanity_z1 = (6, 5, 'right')
    sanity_z2 = (6, 5, 'down')
    print ("Part 1(b) Sanity Check: ")
    print("\nMarginal for hidden state ", sanity_z1, " at time i=1 is: ", marginals[1][sanity_z1])
    print("\nMarginal for hidden state ", sanity_z2, " at time i=1 is: ", marginals[1][sanity_z2])
    '''
    sanity_z30 = (6, 7, 'right')
    print("\nMarginal for hidden state ", sanity_z30, " at time i=30 is: ", marginals[30][sanity_z30])
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

    # number of timesteps
    num_time_steps = len(observations)
    #a distribution of the hidden state that represents the log probability of transition
    #from a factor to a variable in the factor tree/HMM chain
    w_z = [None] * num_time_steps
    #a function to keep track of the most likely previous states for each current
    #timestamp in the forward pass of the algorithm. A list of dicts where the key is a given
    #hidden state for the current timestamp, and the value is a hidden state from the
    #previous timestamp that has the heighest probability to lead to the hidden state
    #denoted by the key
    zPrevMax = [None] * num_time_steps
    #a function to determine the most likely sequence in the backward pass of the
    #algo as we backtrack the HMM chain. A list of estimated hidden states per
    #timestamp
    phi_z = [None] * num_time_steps
    
    #Forward pass of Viterbi Algorithm:
    for i in range (num_time_steps):
        w_z[i] = rover.Distribution()
        zPrevMax[i] = {}
        for current_state in all_possible_hidden_states:
            #Define the observation distribution and the current observation
            obs_distr = observation_model(current_state)
            observation = observations[i]
            #Assume the observation is missing and assign it a unit probability
            P_obs = 1
            #If the observation exists, use the model instead to get the prob
            if observation is not None:
                P_obs = obs_distr.get(observation)
            
            #if the observation distribution didn't yield a non-zero prob for 
            #that observation, that means the hidden state is not really possible
            #based on that observation, so this probability should be zero, however
            #since we are working in the log domain, note that log(0) is undefined
            #so we assign it a very small probability.
            #Also note, we can't skip like we do in F/B because we are taking the
            #sum and not the product
            if P_obs is None:
                P_obs = 1e-10
            #tackle the initialization of the recurrence here instead of outside
            #the loop, it's cleaner code.
            if (i == 0):
                #w_(z0) = log(p(z0)) + log(p(obs0 | z0))
                prior_prob = prior_distribution[current_state]
                #print(prior_prob)
                if prior_prob != 0:
                    w_z[i][current_state] = np.log(prior_prob) + np.log(P_obs)
            else:
                #a hash of the previous states
                prev_states_dict = {}
                previous_states = w_z[i-1]
                #print("There are ",len(previous_states), "previous states")
                for prev_state in previous_states:
                    #Calculate the transition probability P(zi|zi-1)
                    #print("Keeping track of state ", prev_state, " at time ", i-1)
                    trans_distr = transition_model(prev_state)
                    P_trans = trans_distr.get(current_state)
                    #print("P Trance: ", P_trans)
                    if P_trans is None:
                        continue
                    #Calculate the max prev state transition log probability
                    prev_logprob = np.log(P_trans) + w_z[i-1][prev_state]
                    #print("Logprob is ", prev_logprob)
                    #keep track of the probabilities of the previous states
                    prev_states_dict[prev_state] = prev_logprob 
                #store the previous state that is most likely to have gotten us to
                #this current state
                #if not prev_states_dict:
                    #continue
                #print ("Prev state dict:\n", prev_states_dict)
                max_prev_state = max(prev_states_dict, key=prev_states_dict.get)
                zPrevMax[i][current_state] = max_prev_state
                #Calculate w(zi) based on the recurrence relation
                assert(prev_states_dict[max_prev_state] == max(prev_states_dict.values()))
                w_z[i][current_state] = np.log(P_obs) + max(prev_states_dict.values())
    
    save_checkpoint("viterbiDict", zPrevMax)
    save_checkpoint("viterbiW", w_z)
    
    #Backward pass of Viterbi Algorithm.
    #we find the most likely state hidden within the very last timestamp,
    #ie. max z(N-1). From there, we backtrack and find the most likely previous
    #state that would have led to it.
    #Corner case: there could be multiple cases in the state (and this applies for
    #states in previous timestamp) that qualify as the max. Handle this later
    zPrevMax = load_checkpoint("viterbiDict")
    w_z = load_checkpoint("viterbiW")
    wLast = w_z[num_time_steps-1]
    #print(zPrevMax)
    #print (wLast)
    argmax_zLast = max(wLast, key=wLast.get)
    phi_z[num_time_steps-1] = argmax_zLast
    #Now backtrack and find the most likely state in the previous timestamp for
    #that
    for i in range(num_time_steps-2, 0, -1):
        #the state we are backtracking from
        #print(i)
        argmax_zNext = phi_z[i+1]
        #print(argmax_zNext)
        #the tagged state in the previous timestamp that is most likely to have led
        #to the next hidden state.
        max_state = zPrevMax[i+1].get(argmax_zNext)
        #print ("Max leading state:\n", max_state)
        phi_z[i] = max_state
        #print(phi_z)
    
    #print (phi_z)
    estimated_hidden_states = phi_z
    return estimated_hidden_states


if __name__ == '__main__':
   
    enable_graphics = False
    
    missing_observations = True
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
                                 observations)
    print('\n')


   
    timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])
  
    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()
        
