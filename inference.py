import numpy as np
import graphics
import rover

'''
Debug flags
'''
DEBUG = False
PRELOADED = False #use this to speed up Viterbi during debugging/examining results
'''
Utilities to save intermediate files. Useful for Viterbi.
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
    
    #Compute the forward messages
    for i in range(num_time_steps):
        forward_messages[i] = rover.Distribution()
        for state in all_possible_hidden_states:
            innerSum = 0
            #The observation probability distribution
            obs_distr = observation_model(state)
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
            if P_obs_given_hidden is None:
                continue
            #Handle the initialization of the recurrence relation. 
            if i == 0:
                #alpha(z0) = P(z0) * P(obs|z0)
                if prior_distribution[state] is not None:
                    forward_messages[0][state] = prior_distribution[state] * P_obs_given_hidden
            else:
                for previous_state in (forward_messages[i-1]):
                    alpha_prev = (forward_messages[i-1])[previous_state]
                    P_trans = transition_model(previous_state)[state]
                    if (P_trans == 0):
                        continue
                    innerSum = innerSum + (alpha_prev * P_trans)
                forward_messages[i][state] = P_obs_given_hidden * innerSum
        forward_messages[i].renormalize()
    
    #Compute the backward messages
    for i in range(num_time_steps-1, -1, -1):
        backward_messages[i] = rover.Distribution()
        #Initialize the B(zn-1) for the recurrence relation
        if i == num_time_steps - 1:
            for state in all_possible_hidden_states:
                backward_messages[i][state] = 1
        else:
            for next_state in (backward_messages[i+1]):
                #The observation probability distribution
                obs_distr = observation_model(next_state)
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
                prod = backward_messages[i+1][next_state] * obs_prob
                for state in all_possible_hidden_states:
                    trans_prob = transition_model(state)[next_state]
                    if (trans_prob == 0):
                        continue
                    backward_messages[i][state] += (prod * trans_prob)
        backward_messages[i].renormalize()

    #Compute the marginals
    for i in range(num_time_steps):
        marginals[i] = rover.Distribution()
        for z_i in all_possible_hidden_states:
            prod = forward_messages[i][z_i] * backward_messages[i][z_i]
            if prod == 0:
                #report only non-zero probabilities in the distribution
                continue
            marginals[i][z_i] = prod
        marginals[i].renormalize()
    
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
    phi_z = [None] * num_time_steps
    #a function to determine the most likely sequence in the backward pass of the
    #algo as we backtrack the HMM chain. A list of estimated hidden states per
    #timestamp
    zTilde_max = [None] * num_time_steps
    
    #Forward pass of Viterbi Algorithm. Note: this takes a minute or two to run
    #so if you want to quickly get the result after running it once, just turn
    #the preloaded flag on.
    if not PRELOADED:
        for i in range (num_time_steps):
            w_z[i] = rover.Distribution()
            phi_z[i] = {}
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
                #tackle the initialization of the recurrence
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
                        trans_distr = transition_model(prev_state)
                        P_trans = trans_distr.get(current_state)
                        if P_trans is None:
                            continue
                        #Calculate the max prev state transition log probability
                        prev_logprob = np.log(P_trans) + w_z[i-1][prev_state]
                        #keep track of the probabilities of the previous states
                        prev_states_dict[prev_state] = prev_logprob 
                    #store the previous state that is most likely to have gotten us to
                    #this current state
                    #print ("Prev state dict:\n", prev_states_dict)
                    max_prev_state = max(prev_states_dict, key=prev_states_dict.get)
                    phi_z[i][current_state] = max_prev_state
                    #Calculate w(zi) based on the recurrence relation
                    assert(prev_states_dict[max_prev_state] == max(prev_states_dict.values()))
                    w_z[i][current_state] = np.log(P_obs) + max(prev_states_dict.values())
        
        save_checkpoint("viterbiDict", phi_z)
        save_checkpoint("viterbiW", w_z)    
    
    #Backward pass of Viterbi Algorithm.
    #we find the most likely state hidden within the very last timestamp.    
    if PRELOADED:
        phi_z = load_checkpoint("viterbiDict")
        w_z = load_checkpoint("viterbiW")    
    wLast = w_z[num_time_steps-1]
    argmax_zLast = max(wLast, key=wLast.get)
    zTilde_max[-1] = argmax_zLast
    #Now backtrack and find the most likely state in the previous timestamp for
    #that
    for i in range(num_time_steps-2, -1, -1):
        #the state we are backtracking from
        #print(i)
        argmax_zNext = zTilde_max[i+1]
        #print(argmax_zNext)
        #the tagged state in the previous timestamp that is most likely to have led
        #to the next hidden state.
        max_state = phi_z[i+1].get(argmax_zNext)
        #print ("Max leading state:\n", max_state)
        zTilde_max[i] = max_state    

    #print (phi_z)
    estimated_hidden_states = zTilde_max
    #print(len(estimated_hidden_states))
    #print(estimated_hidden_states)
    
    return estimated_hidden_states

def compute_errors(marginals, estimated_states, hidden_states):
    
    """
    Inputs
    ------
    marginals: marginal distributions of z_i for each time stamp
    estimated_states: the most likely sequence obtained from the Viterbi Algorithm
    hidden_states: the true sequence of hidden states
    
    Output
    ------
    A tuple of error probabilities for Viterbi and F/B respectively
    """
    num_time_steps = len(hidden_states)
    correct_viterbi = 0
    correct_FB = 0
    
    for i in range(num_time_steps):
        if hidden_states[i] != marginals[i].get_mode():
            if DEBUG:
                print("Time step ", i)
                print("True hidden state: ", hidden_states[i])
                print("Estimated max marginal: ", marginals[i].get_mode())
        else:
            correct_FB += 1
        
        if hidden_states[i] != estimated_states[i]:
            if DEBUG:
                print("Time step ", i)
                print("True hidden state: ", hidden_states[i])
                print("Estimated Viterbi: ", estimated_states[i])
        else:
            correct_viterbi += 1
     
    if DEBUG:
        print ("Total correct for Viterbi:", correct_viterbi)
        print ("Total correct for F/B", correct_FB)
    
    PErr_Viterbi = 1 - (correct_viterbi/num_time_steps)
    PErr_FB = 1 - (correct_FB/num_time_steps)
    
    error_probabilities = (PErr_Viterbi, PErr_FB)
    
    return error_probabilities

def detect_invalid_sequence(marginals, transition_model):

    """
    Inputs
    ------
    marginals: marginal distributions of z_i for each timestamp
    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state 
    
    Output
    ------
    A tuple consisting of the first invalid (i, z_i and z_(i+1))
    """
    num_time_steps = len(marginals)
    for i in range(num_time_steps-1):
        max_zi = marginals[i].get_mode()
        max_znext = marginals[i+1].get_mode()
        #The transition probability P(zi+1 | zi)
        P_trans = transition_model(max_zi).get(max_znext)
        #If that transition couldn't happen, it wouldn't exist in the transition model
        if P_trans is None:
            return (i, max_zi, max_znext)
    
    return (None, None, None)

if __name__ == '__main__':
   
    enable_graphics = True
    
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
    if missing_observations:
        timestep = 30
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
    
    print('\n')
    
    #Compute the error probabilities:
    Perr_Vit, Perr_FB = compute_errors(marginals, estimated_states, hidden_states)
    print ("Error Probability for Viterbi = ", round(Perr_Vit, 2))
    print ("Error Probability for F/B = ", round(Perr_FB, 2))
    
    print("\n")
    #Detect and invalid sequence in the marginal distribution
    first_invalid_sequence = detect_invalid_sequence(marginals, rover.transition_model)
    print("The first invalid transition segment occurs at time step i = ", first_invalid_sequence[0], \
          "where z_i: ", first_invalid_sequence[1], " and z_i_plus_1: ", first_invalid_sequence[2])
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()
        
