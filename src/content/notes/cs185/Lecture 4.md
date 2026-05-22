# Lecture 4: Reinforcement Learning Basics
## The Markov Decision Process
The **Markov Decision Process (MDP)** is a way of describing reinforcement learning problems.

- Markov: for the Markov property
- Decision: because the model makes decisions

> [!info]
> For now, we will assume $\mathbf{o}_{t}=\mathbf{s}_{t}$, i.e. our policy is $\pi_{\theta}(\mathbf{a}_{t}\mid \mathbf{s}_{t})$.

The critical difference between imitation and reinforcement learning is the lack of demonstration training examples for the model to learn from. Instead, in an MDP, we're provided a **reward function** $r(\mathbf{s}_{t},\mathbf{a}_{t})$, which represents the (state, action) pairs that are preferred. In fact, an MDP may be defined wholly by the following parameters.

- States $\mathbf{s}_{t}$.
- Actions $\mathbf{a}_{t}$.
- Rewards $r(\mathbf{s}_{t},\mathbf{a}_{t})$.
- Probabilities $p(\mathbf{s}_{t+1}\mid \mathbf{s}_{t},\mathbf{a}_{t})$.

> [!info] Markov Chains
> Markov chains originated as a representation for *dynamical states* that satisfied the Markov property, i.e. $\mathbf{s}_{t+1}\perp \mathbf{s}_{t-1}\mid \mathbf{s}_{t}$. It may be represented by $\mathcal{M}=\{ \mathcal{S},\mathcal{T} \}$, where $\mathcal{S}$ is the *state space* and $\mathcal{T}$ is the *transition operator*, which describes the linear operator applied to some vector $\mu_{t}$, for which entry $i$ describes the probability of being in the $i$th state at time $t$, to produce $\mu_{t+1}$, i.e. $\mu_{t+1}=\mathcal{T}\mu_{t}$. Note that $\mathcal{T}_{i,j}=p(\mathbf{s}_{t+1}=i\mid \mathbf{s}_{t}=j)$. Markov chains, of course, inspired MDPs.

More formally, we extend Markov chains to produce MDPs characterized by $\mathcal{M}=\{ \mathcal{S},\mathcal{A},\mathcal{T},r \}$, where

- States $\mathbf{s}\in \mathcal{S}$.
- Actions $\mathbf{a}\in \mathcal{A}$.
- Transition operator $\mathcal{T}$.
- Reward function $r:\mathcal{S}\times \mathcal{A}\to \mathbb{R}$.

> [!question] MDP $\to$ Markov Chain?
> Given a policy $\pi_{\theta}(\mathbf{a}_{t}\mid \mathbf{s}_{t})$, it is possible to turn an MDP into a Markov chain. It suffices to define the states of the Markov chain to be the set of all $(\mathbf{s},\mathbf{a})\in \mathcal{S}\times A$ pairs and define the transition operator using the policy, i.e. $p(\mathbf{s}_{t+1},\mathbf{a}_{t+1}\mid \mathbf{s}_{t},\mathbf{a}_{t})=p(\mathbf{s}_{t+1}\mid \mathbf{s}_{t},\mathbf{a}_{t})\pi_{\theta}(\mathbf{a}_{t+1}\mid \mathbf{s}_{t+1})$.

We note that **partially observed** MDPs, i.e. $\mathbf{o}_{t}\neq \mathbf{s}_{t}$, we must extend this definition slightly, i.e. $\mathcal{M}=\{ \mathcal{S,A,O,T,E},r \}$, where

- States $\mathbf{s}\in \mathcal{S}$.
- Actions $\mathbf{a}\in \mathcal{A}$.
- Observations $\mathbf{o}\in \mathcal{O}$
- Transition operator $\mathcal{T}$.
- Emission operator $\mathcal{E}$.
- Reward function $r:\mathcal{S}\times \mathcal{A}\to \mathbb{R}$.

This can also be written as a Markov chain; for now, though, we will continue will our assumption that $\mathbf{o}_{t}=\mathbf{s}_{t}$, i.e. **fully observed** MDPs.
### Some Useful Concepts
For a **finite horizon** RL problem with policy $\pi_{\theta}$, we can write that the probability of any sequence of states and actions is
$$
p_{\theta}(\mathbf{s}_{1},\mathbf{a}_{1},\dots ,\mathbf{s}_{H},\mathbf{a}_{H}) = p(\mathbf{s}_{1})\prod_{t=1}^{H} \pi_{\theta}(\mathbf{a}_{t}\mid \mathbf{s}_{t})p(\mathbf{s}_{t+1}\mid \mathbf{s}_{t},\mathbf{a}_{t})
$$
by the chain rule of probability. $p_{\theta}(\mathbf{s}_{1},\mathbf{a}_{1},\dots,\mathbf{s}_{H},\mathbf{a}_{H})$ is known as the **trajectory distribution**. Another notable distribution is the **state marginal** $p_{\theta}(\mathbf{s}_{t})$, which describes the probability of being in state $\mathbf{s}_{t}$ at time step $t$ for some policy $\pi_{\theta}$. Critically, both probabilities, while having very long, complex formulas, are easy to compute; the trajectory distribution is derived by simply simulating the MDP (just run the Markov chain), while the state marginal is computed by just simulating the policy in the real world.
## The Objective of RL
Simple—choose a policy that maximizes total expected reward over the entire horizon.
$$
\theta^{*} = \underset{\theta}{\arg\max}\ \mathbb{E}_{\tau\sim p_{\theta}(\tau)}\left[ \sum_{t} r(\mathbf{s}_{t},\mathbf{a}_{t}) \right]
$$
where $\tau$ is the sequence/trajectory of states and actions and $p_{\theta}(\tau)=p_{\theta}(\mathbf{s}_{1},\mathbf{a}_{1},\dots,\mathbf{s}_{H},\mathbf{a}_{H})$, the trajectory distribution.

Viewing it as an MDP in Markov chain form, we can derive that the total expected reward is
$$
\mathbb{E}[\sum_{t=1}^{H}r(\mathbf{s}_{t},\mathbf{a}_{t})]=\left[ \sum_{t=1}^{H} \mathcal{T}_{\theta}^{t-1}\mu_{1} \right]^{T} \vec{r}
$$
where $\vec{r}$ is the vector of rewards for $r(\mathbf{s},\mathbf{a}),\ \forall (\mathbf{s},\mathbf{a})\in \mathcal{s}\times \mathcal{a}$, $\mu_{1}$ is the first state, and $\mathcal{T}_{\theta}$ is the transition probabilities according to the policy $\pi_{\theta}$. In practice, of course, this is never computed this way, but it does give a nice, concise representation of the total expected reward.

But, what if the horizon $H$ is **infinite**? Typically, the total expected reward is represented as $\frac{1}{H}\mathbb{E}[\dots]$ so that it does not approach infinity as well. Using this expression does, in fact, behave well. In particular, provided the underlying Markov chain is

- *Ergodic*: $\forall(\mathbf{s}_{1},\mathbf{s}_{2})\in \mathcal{S}\times \mathcal{S}$, it's possible to reach $\mathbf{s}_{2}$ from $\mathbf{s}_{1}$.
- *Aperiodic*: the system $\mu$ does not return to a state periodically for some fixed interval, i.e. we do not have $\mu_{t}=\mu_{t+k}=\mu_{t+2k}=\dots$ for some $k$.

the terms from the infinite sum, past some sufficiently large threshold $T$, are selected from a **stationary distribution**. This is important because a stationary distribution has the property that
$$
\bar{\mu}=\mathcal{T}_{\theta}\bar{\mu}
$$
which allows us to solve for $\bar{\mu}$, as it implies
$$
(\mathcal{T}_{\theta}-\mathbf{I})\bar{\mu}=0
$$
or that $\bar{\mu}$ is an eigenvector of $\mathcal{T}_{\theta}$ with eigenvalue $1$.
### Expectations and Stochasticity
One critical underlying principle of RL is that we care about *expectations on probabilities*. Even with a discontinuous problem description (e.g. $r(\text{go left})=1$ while $r(\text{go right})=-1$), the expectation $\mathbb{E}_{\pi_{\theta}}[r(\mathbf{x})]$ remains differentiable in $\theta$ because it is being differentiated *w.r.t policy parameters* (which induces probabilities), not states. This allows us to use the same gradient-based optimization techniques widely popularized in machine learning.
## Anatomy of RL Algorithms
The simple cycle is

1. Generate samples (i.e. run policy)
2. Fit a model/estimate return
3. Improve policy

and repeat.

Notably, the relative cost and complexity of these steps differ between algorithms. 

1. Sample generation is cheap for problems that can be simulated, but expensive for those that must be run in real-time
2. Fast for simple gradients, expensive for large model-based methods
3. Again, fast for simple gradients, expensive for large model-based methods with backpropagation
## Value Functions and $Q$-Functions
To calculate total expected reward for each (state, action) pair, we can define a function $Q^{\pi}(\mathbf{s}_{t},\mathbf{a}_{t})$ that is effectively equivalent to the reward $r(\mathbf{s}_{t},\mathbf{a}_{t})$ plus the total expected reward given that the $t$th (state, action) pair is $(\mathbf{s}_{t},\mathbf{a}_{t})$. This should be intuitive, hopefully, and helps us quantify this notion of considering *future rewards* in maximizing total expected reward into one function, the **$Q$-function**. Formally,
$$
Q^{\pi}(\mathbf{s}_{t},\mathbf{a}_{t})=\sum_{t'=t}^{T} \mathbb{E}_{\pi_{\theta}}[r(\mathbf{s}_{t'},\mathbf{a}_{t'})\mid \mathbf{s}_{t},\mathbf{a}_{t}]
$$
A **value function** is a related concept that is based only on the state, i.e. $V^{\pi}(\mathbf{s}_{t})=\sum_{t'=t}^{T}\mathbb{E}_{\pi_{\theta}}[r(\mathbf{s}_{t'},\mathbf{a}_{t'})\mid \mathbf{s}_{t}]$ represents the total expected reward given that the $t$th state is $\mathbf{s}_{t}$. Notably, $\mathbb{E}_{\mathbf{s}_{1}\sim p(\mathbf{s}_{1})}[V^{\pi}(\mathbf{s}_{1})]$ is the RL objective! Also, note that
$$
V^{\pi}(\mathbf{s})=\mathbb{E}[Q^{\pi}(\mathbf{s},\mathbf{a})]
$$
So, how can we use $Q$-functions and value functions? Here are some important ideas.

1.  If $Q^{\pi}(\mathbf{s},\mathbf{a})$ is known, we can simply set $\pi'(\mathbf{a}\mid \mathbf{s})=1$ for $\mathbf{a}=\underset{\mathbf{a}}{\arg\max}\ Q^{\pi}(\mathbf{s},\mathbf{a})$.
2. Although, we can compute a *policy gradient*: if $Q^{\pi}(\mathbf{s},\mathbf{a})>V^{\pi}(\mathbf{s})$, then $\mathbf{a}$ is a better-than-average action, so modifying $\pi(\mathbf{a}\mid \mathbf{s})$ to improve the probability of selecting $\mathbf{a}$ should improve the policy.
## Types of RL Algorithms
- **Policy gradient** methods compute the gradient of the objective with respect to the parameters. They tend to be simple, but require extremely large datasets.
- **Value-based** methods estimate the value function or $Q$-function of the optimal policy. (No explicit policy).
- **Actor-critic** methods estimate the value function or $Q$-function of the current policy to improve/train the policy. (They train a *critic* policy to improve/train the *actor* policy). They tend to be the most efficient methods.
- **Model-based** methods estimate the transition model, and are quite varied and often domain-specific.
### Model-based RL
The model typically learns $p(\mathbf{s}_{t+1}\mid \mathbf{s}_{t}, \mathbf{a}_{t})$, and has several options for how to use this to improve the policy:

- Use the model to plan (basically backprop for trajectory optimization/related algorithms).
- Backpropagate gradients into the policy.
- Learn a value function (e.g. dynamic programming).
### Value-based RL
The model learns $V(\mathbf{s})$ or $Q(\mathbf{s},\mathbf{a})$ and then sets the policy based on its estimate, i.e. $\pi(\mathbf{s})=\underset{\mathbf{a}}{\arg\max}\ Q(\mathbf{s},\mathbf{a})$.
### Policy gradient RL
The model evaluates the policy by summing the rewards, and then improves by computing the gradient of the objective.
### Actor-Credit
Combination of value-based + policy gradient methods.
## Why so many different RL algorithms?
There have tradeoffs.

- Sample efficiency
- Stability and ease of use

They make different assumptions.

- Stochastic or deterministic systems
- Continuous or discrete problem
- Episodic or infinite horizon

Different things are easy in different settings.

- Easier to represent the policy?
- Easier to represent the model?
### Sample Efficiency
The most important consideration is if the algorithm is **off policy**. An off policy algorithm (e.g. model-based) is able to improve the policy *without* generating new samples from the policy, while an on policy (e.g. policy gradient) algorithm must on policygenerate new samples. Typically, the tradeoff is compute cost vs sample generation cost.
### Stability and Ease of Use
To consider stability/ease of use, you should ask

- Does it converge?
- If so, what does it converge to?
- Does it always converge?

Why does this matter? Because, unlike supervised learning, RL is often not gradient-descent based, and thus does not come with the same guarantees. For instance,

- Value-based algorithms frequently have no guarantees of convergence.
- Model-based algorithms will converge, but won't necessarily converge to a good policy.
- Policy gradient algorithms do converge to good policies because they compute gradients directly on the objective function. (Gradient ascent).
<div style="page-break-after: always;"></div>

