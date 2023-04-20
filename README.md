# Snakes

A minimal environment equipped with reinforcement learning algorithms to train agents to compete in [Snakes](https://en.wikipedia.org/wiki/Snake_(video_game_genre)). Due to its simplicity, this repository is potentially useful for educational purposes and can serve as a starting point to solve more complex scenarios.

## Installation

To run *Snakes*, install the latest master directly from GitHub. For a basic install, run:

```console
git clone https://github.com/kaifishr/Snakes
cd Snakes 
pip3 install -r requirements.txt
```

## Getting Started

Run a training session using a specified learning algorithm:

```console
cd Snakes 
python train.py -a "policy_gradient"
python train.py -a "deep_q_learning"
```

Track important metrics during training with Tensorboard:

```console
cd Snakes 
tensorboard --logdir runs/
```

After training, play Snakes against an agent:

```console
cd Snakes 
python play.py -a deep_q_learning -mn agent_a 
```

## Introduction

### Reinforcement Learning

<p align="center">
<img src="https://www.mathworks.com/help/reinforcement-learning/ug/agent_diagram.png" alt="" width="320" height="">
<figcaption>
<font size="1">
mathworks.com/help/reinforcement-learning/ug/agent_diagram.png 
</font>
</figcaption>
</p>

This section is intended to give a *very* brief introduction to some aspects of reinforcement learning and the algorithms, namely [policy gradients](#policy-gradients) and [deep Q-learning](#deep-q-learning), that are used to train agents to play Snakes.

If the learning task that we try to solve cannot be taken offline or reduced to an abstract task like regression or classification, we can formulate it as a reinforcement learning task. In reinforcement learning, agents interact with an uncertain environment, perform actions, and are continually trained to learn how to correctly interact with a dynamic world. The agent consists of the policy network and a reinforcement learning algorithm such as deep Q-learning or policy gradients.

More specifically, let's use the game of Snakes as an example. The dynamic world or environment is represented by multiple interacting agents (the snakes). An agent observes a **state** that is represented by the configuration of the playing field. An example state with a single snake looking for food look as follows

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

Based on the observed state, the agent performs an **action**. This action causes the environment to transition to a new state. Available actions are the set of allowed moves. Following the action, the environment provides a **reward**. The reward is a scalar value, where higher values are better. The agent's action is based on a **policy**. A policy is a function that maps states (the current observation of the environment) to a probability distribution of the actions to be taken and can be modeled by a neural network whose parameters $\boldsymbol \theta$ are learned.

$$\textbf{action}= \text{policy}(\textbf{state}; \boldsymbol \theta)$$

During the training, the agent interacts with the environment and the selected optimization method, adjusts the agent's policy in order to *maximize the expectation of future rewards*.

It should be noted, that for a given state, the agent's best choice for an action depends only on the current state and may not depend on past states. Considering only the information provided by the current state for the next action is known as a [Markow decision process](https://en.wikipedia.org/wiki/Markov_decision_process) (MDP).

### Snakes Environment

This section describes the details of the Snakes environment as well as the action and reward scheme.

The states that the agent observes, and that is later fed into the policy network, are the positions of other snakes as well as the food. A possible encoding scheme for a game with a single agent, where food is encoded as $2$, the snake's body using $1$ s, and empty space using $O$ s, looks as follows:

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

Based on the present state, an agent selects one out of four actions. These actions are going *left*, *right*, *up* and *down*. Actions are determined by the agent's current policy. Here, the policy is modeled as a neural network with four output neurons. The actions are integer-valued and retrieved by applying the argmax function to the network's output neurons.

Even though there are four actions available, not all moves are allowed (hitting obstacles). If the action is legal, the opponents make their moves, and the agent observes the new state.

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


### Self-play

This framework allows agents to be trained using a self-play training strategy. Instead of an algorithmic player as an opponent, we train two agents at the same time and let them compete against each other. 

Alternatively, we can train an agent several episodes until it beats a weaker version of itself a certain number of times. We then make the trained agent the new opponent and start over again. 

To ensure that agents generalize well, for the approaches described above, it is generally a good idea to work with an ensemble of opponent agents that are sampled at random to compete against each other.


### Policy Gradient

The agent's interactions with the environment over the course of one episode can be considered the unrolling of a computational graph. However, parts of this graph are not differentiable, such as the sampling of actions or the environment, as the environment's underlying computational processes might be unknown.

The Policy Gradient reinforcement learning algorithm ignores the problem of credit assignment as it focuses on the overall performance of the agent's policy after running one episode. 

If an episode ends with a positive (negative) total reward, it is expected that, on average, actions associated with a positive (negative) reward occurred more often during the episode. 

In practice, we unroll an episode and compute the discounted rewards to derive the error signal, for each step taken during the episode, which is then used for the policy gradient descent.

Let's look at the math behind policy gradients. Let $a$ be a single action that was taken during an episode rollout and $r$ the final reward of that episode.

As the policy network (resembles a probability density function / produces probabilities) from which we sample actions, the final reward $r$ can be considered a random variable (whose realization is a probabilistic value). Hence, for identical initial states, the decision process and therefore the final reward might look totally different.

The expected total reward $\mathbb{E}[r(a)]$ is therefore what we want to maximize. In the discrete case, the expected value of the reward is defined as

$$
\mathbb{E}[r(a)] = \sum_{a} p(a) r(a)
$$

where $p(a)$ is the probability of the policy network for action $a$. As we want to maximize the total reward, we take the derivative of the expected reward with respect to the policy network's parameters to derive the main equation in Policy Gradient:

$$
\begin{aligned}
\nabla_{\theta} \mathbb{E}[r(a)] 
&= \nabla_{\theta} \sum_{a} p(a | \theta) r(a)\\
&= \sum_{a} \nabla_{\theta} p(a | \theta) r(a)\\
&= \sum_{a} p(a | \theta) \nabla_{\theta} \log (p(a | \theta)) r(a)\\
&= \mathbb{E} [\nabla_{\theta} \log (p(a | \theta)) r(a)]
\end{aligned}
$$

where we compute the expression $\nabla_{\theta} \log (p(a | \theta))$ using backpropagation.

As an aside, the above derivation uses the log-derivative trick. The derivative of function $y = \log (f(x))$ is $\frac{\partial}{\partial x} y = \frac{1}{f(x)} \frac{\partial}{\partial x} f(x)$. Therefore we can write $f(x) \frac{\partial} {\partial x} \log (f(x)) = \frac{\partial} {\partial x} f(x)$. Applying this to the derivative of the probability for action $a$ we get $\nabla_{\theta} p(a | \theta) = \sum_{a} p(a | \theta) \nabla_{\theta} \log (p(a | \theta))$. Aside end.

The derived expectation can be approximated by averaging $T$ samples by dropping the expectation operator altogether.

$$
\begin{aligned}
\nabla_{\theta} \mathbb{E}[r(a)] 
&= \mathbb{E} [\nabla_{\theta} \log (p(a | \theta)) r(a)]\\
&= \frac{1}{T} \sum_{t} \nabla_{\theta} \log (p(a_{t} | \theta)) r(a_{t})]
\end{aligned}
$$

It should be noted that the gradient estimates are unbiased but high in variance. To reduce the variance, other methods such as [control variates](https://en.wikipedia.org/wiki/Control_variates) or [actor critic](http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_5_actor_critic_pdf).


### Deep Q-Learning

The goal of deep Q-learning is to learn the Q-function represented by a deep neural network. 

For discrete action spaces with $n_{a}$ available actions to choose from ($n_{a} = 9$ actions for Snakes), the Q-function maps a state $s$ to $n_{a}$ outputs predicting the Q-values $Q(s, a)$ for each action. This is in contrast to policy gradients, where the policy-network's output represents a probability distribution over a discrete action space. The policy $\pi$ determining the action to take in state $s$, is given by the index associated with the network's maximally activated output neuron.

If the network represents the true Q-function, then it satisfies the Bellman equation

$$Q(s,a) = r(s,a) + \gamma \mathbb{E}[Q(s', \pi(s'))]$$

with policy $\pi$ representing the action that maximizes the expected reward starting with state $s$

$$\pi = \text{argmax}_{a} Q(s, a)$$

As the randomly initialized neural network modeling the Q-function is far from representing the true Q-function, it does not satisfy the Bellman equation, and thus a [temporal difference error](https://en.wikipedia.org/wiki/Temporal_difference_learning) $\epsilon$ exists

$$\epsilon = r(s,a) + \gamma \mathbb{E}[Q(s', \pi(s'))] - Q(s,a)$$

The objective is to learn the parameters of the neural network representing the Q-function by minimizing the temporal difference error over the entire trajectory using the mean squared error

$$MSE = \frac{1}{2} \sum_{t=1}^{T} \epsilon_{t}^{2}$$

The Q-function considered here represents a deterministic policy. A deterministic policy results in always performing the same actions given a fixed initial state, preventing exploration. To allow for some exploration, the selection of an action should contain some randomness, using, for example, epsilon-greedy sampling for a non-deterministic trajectory.

Another difference to policy gradients is that Q-learning separates exploration used to learn the Q-function from exploitation of the Q-function once learning is finished. In contrast, policy gradients follow the current policy.

A benefit of deep Q-learning is the possibility of updating the policy's parameters directly after each step (taking an action and observing the reward). This is in contrast to policy gradients, where the events of the decision process (states, actions, and rewards) are recorded for an entire episode and the final reward has been observed before the policy update can be performed.


## References

* [A Crash Course on Reinforcement Learning](https://arxiv.org/abs/2103.04910)


## TODO

- Encode length of snake. [1, 2]
- Set-up engine for single agent.
- Turn on / off growth.
- Implement adaptive epsilon decay rate for deep q-learning.
- Add Boltzmann exploration and epsilon-greedy sampling.


## License

MIT
