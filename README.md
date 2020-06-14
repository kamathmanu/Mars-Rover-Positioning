# Positional Inference of a Mars Rover over time using Graphical Hidden Markov Models

## Background Information

When tracking the position of a robot/rover over a terrain, it is likely that the *exact* position of the rover is not known; the best we might be able to do is to capture some noisy observations of the rover's position. However, it is still possible to model its movement by *inferring its movement*, by thinking of the rover as a Hidden Markov Model (HMM). 

For this case, we simplify the modelling by considering the terrain as a 2D coordinate grid with dimensions 12 x 8. 

The rover's position at time `i = 0,1,2, . . .` is modeled as a random vector : `(x<sub>i</sub>, y<sub>i</sub>) ∈ {0,1, . . . ,11} ×{0,1, . . . ,7}`. The movement of the rover is as follows: At each time step, it makes one of the five actions: it stays put, goes left, goes up, goes right, or goes down.


We note that the movement of the rover <ins>depends on both its current and previous position</ins>. We estimate the probabilities of its movement as follows:

- Assuming the robot isn't at the boundary of the grid, if the robot was moving, the robot is likely to move (in the same direction) with probability 0.9, and stays in the same location with probability 0.1.
- If the robot was stationary in the previous step, then all of the five actions are equally likely (probability 0.2).
- If, however, the robot is at the boundary, it has a an intelligent sensing mechanism that prevents it from going off the terrain, and adjusts the set of possible actions based on this. 

We model the rover’s hidden state *z<sub>i</sub>* at time *i* as a super variable that includes both the rover’s location *(x<sub>i</sub>, y<sub>i</sub>)* and its most recent action *a<sub>i</sub>* , i.e. `z<sub>i</sub> = ((x<sub>i</sub> , y<sub>i</sub> ), a<sub>i</sub> )`, where a<sub>i</sub> is a random variable that takes the value from `{stay, left, up, right, down}` and corresponds to the action taken in the <ins>previous</ins> timestep.

### Why does this matter?

<ins>**Using the Forward-Backward Algorithm**</ins>, it is possible to estimate the most likely *z<sub>N</sub>* by computing the marginal distribution `P(z |(x ,y ),...,(x ,y )) for i = 1,2,...,N`.

<ins>**Using the Viterbi Algorith**</ins>, we can infer the most likely *sequence of steps taken by the rover* for `i = 1,2,...,N`.

For more background info on the actual mathematics behind these algorithms and the fundamentals of HMMs, one is advised to refer to Murphy's *Machine Learning - A Probabilistic Perspective* (§17.4), or Bishop's *Pattern Recognition and Machine Learning* (§8.4). 

## Running the code

This code requires Python3 to run. To run, simply `python3 inference.py`.

