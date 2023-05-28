# Snakes

A minimal environment equipped with reinforcement learning algorithms to train agents to compete in [Snakes](https://en.wikipedia.org/wiki/Snake_(video_game_genre)). Due to its simplicity, this repository is potentially useful for educational purposes and can serve as a starting point to solve more complex scenarios and to test reinforcement learning algorithms.

## Installation

To run *Snakes*, install the latest master directly from GitHub. For a basic 
install, run:

```console
git clone https://github.com/kaifishr/Snakes
cd Snakes 
pip3 install -r requirements.txt
```

## Getting Started

Run a training session using a specified learning algorithm:

```console
cd Snakes 
python train.py --algorithm policy_gradient
python train.py --algorithm deep_q_learning
```

Track training progress with Tensorboard:

```console
cd Snakes 
tensorboard --logdir runs/
```

Watch a trained agent play Snakes:

```console
cd Snakes 
python play.py --mode agent --model-name 1
```

## Introduction

### Reinforcement Learning

The interaction of an agent with an uncertain environment and the selection of
the right actions to maximize a reward are difficult and sometimes unfeasible 
to learn by formulating them as an regression or classification task. This is
where reinforcement learning comes into play to continually train an agent to
learn how to correctly interact with a dynamic world.

This repository comes with two basic reinforcement learning algorithms, namely 
[policy gradients](#policy-gradients) and [deep Q-learning](#deep-q-learning), 
that are used to train agents to play Snakes. The agent consists of a policy
network and the reinforcement learning algorithm used to train the network.

In the game of Snakes, the dynamic world, also referred to as the environment,
is represented by the playing field and possibly other agents (in this case
other snakes) looking for food. An agent observes a **state** represented by 
the current configuration of the playing field. An example state with a single 
snake looking for food look as follows

$$
\begin{aligned}
&\begin{array}{|ccccc|}
\hline
\times & \circ & \circ & \circ & \circ \\
. & . & . & . & \circ \\
. & \circ & \circ & . & \circ \\
. & \circ & . & . & \circ \\
. & \circ & \circ & \circ & \circ \\
\hline
\end{array}
\end{aligned}
$$

with $\circ$ and $\times$ representing the snake's body and food, respectively. 

Based on the observed state, the agent performs an **action**. This action 
causes the environment to transition to a new state. In the case of Snakes, the
available actions is the set of possible movement directions ($\uparrow$, 
$\downarrow$, $\leftarrow$, $\rightarrow$).

The agent's actions are based on a **policy**, a function that maps states 
(the current observation of the environment) to a discrete probability 
distribution of possible actions. This function can be modeled by a neural 
network whose parameters $\boldsymbol \theta$ are learned.

$$\textbf{action}= \text{policy}(\textbf{state}; \boldsymbol \theta)$$

As an aside, it should be noted, that for a given state, the agent's best 
choice for an action depends only on the current state and may not depend on 
past states. Considering only the information provided by the current state for 
the next action is known as a [Markow decision process](https://en.wikipedia.org/wiki/Markov_decision_process). Aside end.

The action is then followed by a **reward** provided by the environment. The 
reward is a scalar value (higher values are better). During the training, the 
agent interacts with the environment and the selected optimization method, 
adjusts the agent's policy in order to *maximize the expectation of future 
rewards*.

### Snakes Environment

The *Snakes* environment is a grid-like structure that is completely observed by the agent. The position of food and the agent itself on the playing field is encoded into the playing field and represents the state an agent can observe.

#### Encoding

A possible encoding scheme for a game with a single agent, where food is encoded as $2$, the snake's body using $1$ s, and empty space using $O$ s, looks as follows:

$$
\begin{pmatrix}
2 & 1 & 1 & 1 & 1 \\
0 & 0 & 0 & 0 & 1 \\
0 & 1 & 1 & 0 & 1 \\
0 & 1 & 0 & 0 & 1 \\
0 & 1 & 1 & 1 & 1 \\
\end{pmatrix} = 
\begin{aligned}
&\begin{array}{|ccccc|}
\hline
\times & \circ & \circ & \circ & \circ \\
. & . & . & . & \circ \\
. & \circ & \circ & . & \circ \\
. & \circ & . & . & \circ \\
. & \circ & \circ & \circ & \circ \\
\hline
\end{array}
\end{aligned}
$$

However, this encoding scheme has the weakness that it is not clear in which direction the agent is oriented. Possible approaches to fix this problem is to add a color gradient or to subtract the present from the last state to encode where the agent is going.

#### Actions

Based on the present state, an agent selects one out of four actions. These actions are going *left*, *right*, *up* and *down*. Actions are determined by the agent's current policy. Here, the policy is modeled as a neural network with four output neurons. The actions are integer-valued and retrieved by applying the argmax function to the network's output neurons.

Even though there are four actions available, not all moves are allowed (colliding with own body or hitting obstacles such as other agents or the playing fields border). If the action is legal, the opponents make their moves, and the agent observes the new state.

States where the agent finds food come with a reward of +1. Making illegal moves results in a reward of -1. Punishing wrong moves encourages the agent to learn only legal moves over time. All other states (including draws) yield a reward of 0. 

### Policy Network

[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) showed that the use of deep neural networks is a powerful option to represent reinforcement learning models that map states to (a distribution over) actions.

The policy network receives a state difference matrix $\boldsymbol S$ and uses a softmax output layer to return a probability distribution over the four possible actions. 

In order to detect the direction of the moving snakes, the difference of two consecutive frames (or states) are fed into the policy network. That means that in a preprocessing step, we subtract the current from the last frame and feed the difference frame to our network. Using the example from above, we can illustrate this as follows:

$$
\begin{aligned}
\boldsymbol p_{t}
= \text{policy}
\begin{pmatrix}
\boldsymbol S_{t},
\boldsymbol S_{t-1}
; \boldsymbol \theta
\end{pmatrix}
\end{aligned}
$$

$$
\begin{pmatrix}
0.4\\
0.1\\
0.3\\
0.2
\end{pmatrix}
= \text{policy}
\begin{pmatrix}
\begin{pmatrix}
2 & 1 & 1 & 1 & 1 \\
0 & 0 & 0 & 0 & 1 \\
0 & 1 & 1 & 0 & 1 \\
0 & 1 & 0 & 0 & 1 \\
0 & 1 & 1 & 1 & 1
\end{pmatrix}_{t},
\begin{pmatrix}
2 & 0 & 1 & 1 & 1 \\
0 & 0 & 0 & 0 & 1 \\
0 & 1 & 1 & 1 & 1 \\
0 & 1 & 0 & 0 & 1 \\
0 & 1 & 1 & 1 & 1
\end{pmatrix}_{t-1}
; \boldsymbol \theta
\end{pmatrix}
$$

We can choose an action by either choosing the action with the highest probability or by sampling from the output probability distribution.


### Episodic Learning

In a reinforcement learning setting, an agent can theoretically learn a task in an online mode ([see this example](https://arxiv.org/pdf/2208.07860.pdf)), where the agent's policy (the neural network) is continuously updated. However, in practice, this can lead to unpredictable behavior by the agent that is difficult to control.

Instead of updating the agent's policy at every time step, a common approach is to update the policy between episodes. An episode can be defined as a task we want the agent to learn. For this project, one episode is a game of Snakes, but it can also be the task of [landing a rocket booster autonomously](https://github.com/kaifishr/RocketLander).

During an episode, the agent takes actions according to its current policy and collects the rewards. We then use this information to update the policy's parameters and start a new episode.


### Multi-agent Reinforcement Learning

This framework allows several agents to be trained at the same time and let them compete against each other. To ensure that the agents generalize well, it is generally a good idea to work with an ensemble of opponent agents that are sampled at random to compete against each other.


## TODO

- Allow slow growth.
- Make encoding optional.
- Implement adaptive epsilon decay rate for deep q-learning.
- Add Boltzmann exploration and epsilon-greedy sampling.


## License

MIT
