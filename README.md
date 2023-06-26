# Snake

A minimal environment equipped with reinforcement learning algorithms to train an agent to play [Snake](https://en.wikipedia.org/wiki/Snake_(video_game_genre)). Both the learning algorithms as well as the game engine are implemented from scratch with a focus on simplicity, making it potentially useful for educational purposes and serving as a starting point to solve more complex scenarios, test reinforcement learning algorithms, and other extensions.

## Installation

To run *Snake*, install the latest master directly from GitHub. For a basic 
install, run:

```console
git clone https://github.com/kaifishr/Snake
cd Snake 
pip3 install -r requirements.txt
```

## Getting Started

Run a training session using a specified learning algorithm:

```console
cd Snake 
python train.py --algorithm policy_gradient
python train.py --algorithm deep_q_learning
```

Track training progress with Tensorboard:

```console
cd Snake 
tensorboard --logdir runs/
```

Watch a trained agent play Snake:

```console
cd Snake 
python play.py --mode agent --model-name 1
```

## Solving Snake with Reinforcement Learning

### Reinforcement Learning

The interaction of an agent with an uncertain environment and the selection of the right actions to maximize a reward are difficult and sometimes unfeasible to learn by formulating them as an regression or classification task. This is where reinforcement learning comes into play to continually train an agent to learn how to correctly interact with a dynamic world.

This repository comes with two basic reinforcement learning algorithms, namely [policy gradients](#policy-gradients) and [deep Q-learning](#deep-q-learning), that are used to train an agent to play Snake. The agent consists of a policy network and the reinforcement learning algorithm used to train the network.

In the game of Snake, a dynamic world, also referred to as the environment, is represented by the playing field, food that is randomly positioned inside it, and the agent itself. The agent observes a **state** represented by the environment's current configuration. An example state  might look as follows

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

Based on the observed state, the agent performs an **action**. This action causes the environment to transition to a new state. In the case of Snake, the available actions is the discrete set of possible movement directions ($\uparrow$, $\downarrow$, $\leftarrow$, $\rightarrow$).

The agent's actions are based on a **policy**, a function that maps states (the current observation of the environment) to a discrete probability distribution of possible actions. This function can be modeled by a neural network whose parameters $\boldsymbol \theta$ are learned.

$$\textbf{action}= \text{policy}(\textbf{state}; \boldsymbol \theta)$$

As an aside, it should be noted, that for a given state, the agent's best choice for an action depends only on the current state and may not depend on past states. Considering only the information provided by the current state for the next action is known as a [Markow decision process](https://en.wikipedia.org/wiki/Markov_decision_process). Aside end.

The action is then followed by a **reward** provided by the environment. The reward is a scalar value (higher values are better). During the training, the agent interacts with the environment and the selected optimization method adjusts the agent's policy in order to *maximize the expectation of future rewards*.

### Snake Environment

The *Snake* environment is a grid-like structure that is completely observed by the agent. The position of food and the agent itself on the playing field is encoded into the playing field and represents the state an agent can observe.

#### Encoding

A possible encoding scheme for a game with a single agent, where food is encoded as $2$, the snake's body using $1$ s, and empty space using $O$ s, looks as follows:

$$
\begin{pmatrix}
-1 & 1 & 1 & 1 & 1 \\
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

However, this encoding scheme has the weakness that it is not clear in which direction the agent is moving, making it hard to train an agent. In order to detect the direction of the moving snake, either a sequence of past frames can be fed into the network or a color gradient can be added to the snake's encoding. Alternatively, the difference of two consecutive frames (or states) can be fed into the policy network.

#### Actions

Based on the present state, an agent selects one out of four actions. These actions are going *left*, *right*, *up* and *down*. Actions are determined by the agent's current policy. Here, the policy is modeled as a neural network with four output neurons. The actions are integer-valued and retrieved by applying the argmax function to the network's output neurons.

Even though there are four actions available, not all moves are allowed (colliding with one's own body or hitting the playing field border). If the action is legal, the agent observes the new state the environment has transitioned to.

States where the agent finds food come with a reward of +1. Illegal moves result in a reward of -1. Punishing wrong moves encourages the agent to learn only legal moves over time. All other states (including draws) yield a reward of 0. 

### Policy Network

[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) showed neural networks are a powerful option to represent a policy that maps states to (a distribution over) actions. The policy network receives a state $\boldsymbol S$ and returns a distribution over the four possible actions. 

$$
\begin{aligned}
\boldsymbol p_{t}
= \text{policy}
\begin{pmatrix}
\boldsymbol S_{t}
; \boldsymbol \theta
\end{pmatrix}
\end{aligned}
$$

Using the gradient-based encoding scheme from above, the policy network's mapping can be illustrated as follows:

$$
\begin{pmatrix}
0.4\\
0.1\\
0.3\\
0.2
\end{pmatrix}_{t}
= \text{policy}
\begin{pmatrix}
\begin{pmatrix}
-1 & 1.1 & 1.2 & 1.3 & 1.4 \\
0 & 0 & 0 & 0 & 1.5 \\
0 & 2.3 & 2.4 & 0 & 1.6 \\
0 & 2.2 & 0 & 0 & 1.7 \\
0 & 2.1 & 2.0 & 1.9 & 1.8
\end{pmatrix}_{t}
; \boldsymbol \theta
\end{pmatrix}
$$

In case a sequence of frames is used as the input, the mapping can be illustrated as follows:

$$
\begin{pmatrix}
0.4\\
0.1\\
0.3\\
0.2
\end{pmatrix}_{t}
= \text{policy}
\begin{pmatrix}
\begin{pmatrix}
-1 & 1 & 1 & 1 & 1 \\
0 & 0 & 0 & 0 & 1 \\
0 & 1 & 1 & 0 & 1 \\
0 & 1 & 0 & 0 & 1 \\
0 & 1 & 1 & 1 & 1
\end{pmatrix}_{t},
\begin{pmatrix}
-1 & 0 & 1 & 1 & 1 \\
0 & 0 & 0 & 0 & 1 \\
0 & 1 & 1 & 1 & 1 \\
0 & 1 & 0 & 0 & 1 \\
0 & 1 & 1 & 1 & 1
\end{pmatrix}_{t-1}
; \boldsymbol \theta
\end{pmatrix}
$$

This implementation supports both variants and also allows them to be combined.   

If the network's output happens to be a probability distribution (as is the case with policy gradients) and action can be choosen by either selecting the action with the highest probability or by sampling from the output probability distribution.

### Episodic Learning

In a reinforcement learning setting, an agent can theoretically learn a task in an online mode ([see this example](https://arxiv.org/pdf/2208.07860.pdf)), where the agent's policy (the neural network) is continuously updated. However, in practice, this can lead to unpredictable behavior by the agent that is difficult to control.

Instead of updating the agent's policy at every time step, a common approach is to update the policy between episodes. An episode can be defined as a task we want the agent to learn. For this project, one episode is a game of Snake, but it can also be the task of [landing a rocket booster autonomously](https://github.com/kaifishr/RocketLander). During an episode, the agent takes actions according to its current policy and collects the rewards. We then use this information to update the policy's parameters and start a new episode.

## TODO

- Add FIFO buffer for policy gradients.
- Add time stamp to console print.
- Implement adaptive epsilon decay rate for deep q-learning.
- Add Boltzmann exploration and epsilon-greedy sampling.


## License

MIT