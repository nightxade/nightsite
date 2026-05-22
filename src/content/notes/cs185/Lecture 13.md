# Lecture 13: Control as Variational Inference
## Theory
Recall that our "control as inference" derivation from [[Lecture 12]] used the following value functions.
$$
\begin{align*}
Q_{t}(\mathbf{s}_{t},\mathbf{a}_{t}) &= r(\mathbf{s}_{t},\mathbf{a}_{t})+\log \mathbb{E}[\exp(V_{t+1}(\mathbf{s}_{t+1}))] \\
V_{t}(\mathbf{s}_{t}) &= \log \int \exp(Q_{t}(\mathbf{s}_{t},\mathbf{a}_{t})) \,\mathrm{d}\mathbf{a}_{t} 
\end{align*}
$$
recall that, with stochastic environment dynamics, the second term of the $Q$-function expression encourages *excessive optimism* regarding states with very large rewards/values, even if they occur with very small state transition probabilities. We'd like to eliminate this excessive optimism.

Let's first take a step back and consider: why does this occur in the first place?

Our inference problem is $p(\mathbf{s}_{1:T},\mathbf{a}_{1:T}\mid \mathcal{O}_{1:T})$, i.e. what is the probability of the trajectory $\tau$ given that the monkey is being optimal? Through marginalization over the time steps $t$ and conditioning on the state $\mathbf{s}_{t}$, we produce the policy $p(\mathbf{a}_{t}\mid \mathbf{s}_{t},\mathcal{O}_{1:T})$, i.e. what is the probability of the action $\mathbf{a}_{t}$ being chosen in state $\mathbf{s}_{t}$ given the optimality of the monkey? However, we may note that $\mathcal{O}_{t}$ is really just an indicator (*evidence*) of whether or not $\mathbf{a}_{t}$ was high reward or not; thus, this is really asking, "given that you obtained high reward, what was your action probability?" This, intuitively, is a good question to ask—after understanding that we obtained a high reward, we'd like to maximize our action probability for this high reward.

However, when performing inference over the computational graph, the action probabilities are not the only factors; critically, the *state transition probabilities* are involved too. In fact, we can similarly marginalize over $t$ and condition to produce $p(\mathbf{s}_{t+1}\mid \mathbf{s}_{t},\mathbf{a}_{t},\mathcal{O}_{1:T})\neq p(\mathbf{s}_{t+1}\mid \mathbf{s}_{t},\mathbf{a}_{t})$. That is, "given that you obtained high reward, what was your transition probability?" And this is a bad question to ask! This is like, say, training our model to make a decision on whether or not to buy a lottery ticket after they already observed winning the lottery ticket in the future. However, just because it won this time does not mean that buying a lottery ticket is a good action in general—in essence, the model is learning action optimality from the fact that it "lucked out" from hitting rare state transition probabilities, and believes it can optimize this state transition probability to always happen.

> [!question]- Still confused?
> Courtesy of Gemini, who clarified the lecture's idea for me. As you noted, standard inference asks: _"Given that I succeeded, what must have happened?"_ If you condition a probabilistic graphical model on achieving high reward ($\mathcal{O}_{1:T}=1$), the model uses Bayes' rule to update the probability of **all** latent variables that caused that reward. Because both the actions $\mathbf{a}_{t}$​ and the state transitions $\mathbf{s}_{t+1}$ are latent variables leading to the reward, the model updates both.
> 
> This leads to the model becoming "delusional." If it is in a stochastic environment where taking a risky action usually results in death but has a 1% chance of hitting a jackpot, standard inference looks at the jackpot and assumes the 1% chance is actually a certainty. It essentially assumes it can control the environment's dice rolls.

In short, we'd like to phrase the question as "given that you obtained high reward, what was your action probability, given that your transition probability did not change?" More generally, we'd like to find a way to tell the model to maximize the probability of success without changing the environment dynamics. Or, mathematically, we'd like to find another distribution $q(\mathbf{s}_{1:T},\mathbf{a}_{1:T})$ that is close to the posterior $p(\mathbf{s}_{1:T},\mathbf{a}_{1:T}\mid \mathcal{O}_{1:T})$ but has static dynamics $p(\mathbf{s}_{t+1}\mid \mathbf{s}_{t},\mathbf{a}_{t})$ (i.e. the model should not believe it is capable of maximizing state transition probabilities like action probabilities).

Well, we can actually apply **variational inference** to this problem! Let our observed variables be $\mathbf{x}=\mathcal{O}_{1:T}$ and our latent variables be $\mathbf{z}=(\mathbf{s}_{1:T},\mathbf{a}_{1:T})$. We'd like to find a $q(\mathbf{z})$ that approximates $p(\mathbf{z}\mid \mathbf{x})$; in particular, we will choose to define
$$
q(\mathbf{s}_{1:T},\mathbf{a}_{1:T})=p(\mathbf{s}_{1})\prod_{t}p(\mathbf{s}_{t+1}\mid \mathbf{s}_{t},\mathbf{a}_{t})q(\mathbf{a}_{t}\mid \mathbf{s}_{t})
$$
Notably, it's very unusual to include $p(\mathbf{s}_{1})$ and $p(\mathbf{s}_{t+1}\mid \mathbf{s}_{t},\mathbf{a}_{t})$ in the definition of $q$, as, in variational inference, it's typically desirable to make $q$ a simple distribution (for a tractable approximation), and we have no guarantees about the simplicity of those distributions derived from the computational graph. We will immediately see why this is useful, though (induces a lot of cancellation below).

![[control-vi-graphs.png]]

Then, according to the **ELBO** (evidence lower bound)
$$
\begin{align*}
\log p(x) &\geq \mathbb{E}_{\mathbf{z}\sim q(\mathbf{z})}[\log p(\mathbf{x},\mathbf{z})-\log q(\mathbf{z})] \\
\log p(\mathcal{O}_{1:T}) &\geq  \mathbb{E}_{(\mathbf{s}_{1:T},\mathbf{a}_{1:T})\sim q}\Bigg[ \left( \log p(\mathbf{s}_{1})+\sum_{t=1}^{T} \log p(\mathbf{s}_{t+1}\mid \mathbf{s}_{t},\mathbf{a}_{t})+\sum_{t=1}^{T} \log p(\mathcal{O}_{t}\mid \mathbf{s}_{t},\mathbf{a}_{t}) \right) \\
&\qquad \qquad \qquad- \left(\log p(\mathbf{s}_{1})+\sum_{t=1}^{T}\log p(\mathbf{s}_{t+1}\mid \mathbf{s}_{t},\mathbf{a}_{t})+\sum_{t=1}^{T}\log q(\mathbf{a}_{t}\mid \mathbf{s}_{t}) \right)\Bigg] \\
&= \mathbb{E}_{(\mathbf{s}_{1:T},\mathbf{a}_{1:T})\sim q}\left[\sum_{t=1}^{T}\log p(\mathcal{O}_{t}\mid \mathbf{s}_{t},\mathbf{a}_{t})-\log q(\mathbf{a}_{t}\mid \mathbf{s}_{t})\right] \\
&= \mathbb{E}_{(\mathbf{s}_{1:T},\mathbf{a}_{1:T})\sim q}\left[\sum_{t}r(\mathbf{s}_{t},\mathbf{a}_{t})-\log q(\mathbf{a}_{t}\mid \mathbf{s}_{t})\right]
\end{align*}
$$
For the last step, recall that we defined $p(\mathcal{O}_{t}\mid \mathbf{s}_{t},\mathbf{a}_{t})=\exp(r(\mathbf{s}_{t},\mathbf{a}_{t}))$ last lecture. This allows us to subsequently conclude that
$$
\log p(\mathcal{O}_{1:T}) \geq \sum_{t}\mathbb{E}_{(\mathbf{s}_{t},\mathbf{a}_{t})\sim q}[r(\mathbf{s}_{t},\mathbf{a}_{t})+\mathcal{H}(q(\mathbf{a}_{t}\mid \mathbf{s}_{t}))]
$$
This looks pretty similar to what we've seen before; in fact, it's just our regular RL policy gradient objective plus the entropy of the action distribution! Crucially, though, the only thing within the expectation that is not fixed relative to $q$ (since the expectation is computed over $q$) is the action probabilities! Our choice to hardcode the environment dynamics $p(\mathbf{s}_{t})$ and $p(\mathbf{s}_{t+1}\mid \mathbf{s}_{t},\mathbf{a}_{t})$ into $q(\tau)$ cancels with the existing probabilities in $p(\tau,\mathcal{O}_{1:T})$. Thus, our model won't try and modify our state transition probabilities to maximize $p(\mathcal{O}_{1:T})$; it's forced to optimize only $q(\mathbf{a}_{t}\mid \mathbf{s}_{t})$, which is the only aspect the model actually controls.

Just one final note to close this off: we still need to solve for $q(\mathbf{a}_{t}\mid \mathbf{s}_{t})$ for all $t$. We can recurse backwards, like we did for backwards messages. We first compute the base case
$$
\begin{align*}
q(\mathbf{a}_{T}\mid \mathbf{s}_{T}) &= \underset{q}{\arg\max}\ \mathbb{E}_{\mathbf{s}_{T}\sim q(\mathbf{s}_{T})}[\mathbb{E}_{\mathbf{a}_{T}\sim q(\mathbf{a}_{T}\mid \mathbf{s}_{T})}[r(\mathbf{s}_{T},\mathbf{a}_{T})]+\mathcal{H}(q(\mathbf{a}_{T}\mid \mathbf{s}_{T}))] \\
&= \frac{\exp(r(\mathbf{s}_{T},\mathbf{a}_{T}))}{\int \exp(r(\mathbf{s}_{T},\mathbf{a})) \,\mathrm{d}a, } = \exp(Q(\mathbf{s}_{T},\mathbf{a}_{T})-V(\mathbf{s}_{T}))
\end{align*}
$$
and then, with many steps of mathematical derivation, we can derive
$$
q(\mathbf{a}_{t}\mid \mathbf{s}_{t})=\exp(Q(\mathbf{s}_{t},\mathbf{a}_{t})-V(\mathbf{s}_{t}))
$$
where $Q(\mathbf{s}_{t},\mathbf{a}_{t})=r(\mathbf{s}_{t},\mathbf{a}_{t})+\mathbb{E}[V_{t+1}(\mathbf{s}_{t+1})]$. Notably, it is now a regular Bellman backup equation; it's no longer optimistic.

And thus, we can compute our value functions backwards through time with our recursive expressions. For $t=T-1$ to $1$,
$$
\begin{align*}
Q_{t}(\mathbf{s}_{t},\mathbf{a}_{t}) &= r(\mathbf{s}_{t},\mathbf{a}_{t})+\mathbb{E}[V_{t+1}(\mathbf{s}_{t+1})] \\
V_{t}(\mathbf{s}_{t}) &= \log \int \exp(Q_{t}(\mathbf{s}_{t},\mathbf{a}_{t})) \,\mathrm{d}\mathbf{a}_{t} 
\end{align*}
$$
and we now have a **soft value iteration** algorithm:

1. Set $Q(\mathbf{s},\mathbf{a})\leftarrow r(\mathbf{s},\mathbf{a})+\mathbb{E}[V(\mathbf{s}')]$.
2. Set $V(\mathbf{s})\leftarrow\text{soft max}_{\mathbf{a}}\ Q(\mathbf{s},\mathbf{a})$.

Various modifications like *discounting* and *temperature* may also be added.
## Maximum Entropy RL Algorithms
> [!question] Why use soft max?
> The easy answer is that it encourages more exploration, though this does not capture the entire impact of the soft max. The true answer is that it encourages robustness to inaccurately specified MDPs, i.e. the addition of the entropy term $\mathcal{H}(q)$ encourages the model to seek high rewards while acting as randomly as possible, eventually placing the learned policy in an action space that is robust to perturbations. In reality, though, the real answer is that it just works well in practice ;)
### $Q$-Learning with soft optimality
Standard $Q$-learning uses the *hard max* over the $Q$ values to determine the value function, i.e. $V(\mathbf{s}')=\max_{\mathbf{a}'}Q_{\phi}(\mathbf{s}',\mathbf{a}')$. For **soft $Q$-learning**, we just change the hard max to soft max. Ultimately, this not a very popular algorithm because, for small action spaces, the hard max works just fine.
### Policy Gradient with soft optimality
Recall that the control with variational inference strategy ended up with a policy gradient objective with an added action entropy term. The modification to policy gradient is precisely the same, just define $J(\theta)=\sum_{t}\mathbb{E}_{(\mathbf{s}_{t},\mathbf{a}_{t})\sim \pi_{\theta}}[r(\mathbf{s}_{t},\mathbf{a}_{t})+\mathcal{H}(\pi_{\theta}(\mathbf{a}_{t}\mid \mathbf{s}_{t}))]$. This is very commonly applied, because it's simple to add (entropy usually has a closed form expression for most policy classes) and policy gradient often collapses to a deterministic, non-exploratory policy too soon during training.
### Soft Actor-Critic
This is the most well-known application of soft optimality. We modify two things:
$$
\begin{align*}
y_{i} &= r(\mathbf{s}_{i},\mathbf{a}_{i})+\gamma \mathbb{E}_{\mathbf{a}'\sim \pi_{\theta}(\mathbf{a}'\mid \mathbf{s}_{i}')}[\hat{Q}_{\phi}^{\pi}(\mathbf{s}_{i}',\mathbf{a}')+\underbrace{ \mathcal{H}(\pi_{\theta}(\mathbf{a}'\mid \mathbf{s}_{i}')) }_{ \text{new} }] \\
J(\theta) &= \sum_{i}\mathbb{E}_{\mathbf{a}\sim \pi_{\theta}(\mathbf{a}\mid \mathbf{s}_{i})}[\hat{Q}_{\phi}^{\pi}(\mathbf{s}_{i},\mathbf{a})+\underbrace{ \mathcal{H}(\pi_{\theta}(\mathbf{a}\mid \mathbf{s}_{i})) }_{ \text{new} }]
\end{align*}
$$
where you may recall that the $y_{i}$ are the target values used to train the critic $\hat{Q}_{\phi}^{\pi}$. Note that, conventionally, the entropy for the $Q$-function is usually estimated, i.e.
$$
y_{i}\approx r(\mathbf{s}_{i},\mathbf{a}_{i})+\gamma \mathbb{E}_{\mathbf{a}'\sim \pi_{\theta}(\mathbf{a}'\mid \mathbf{s}_{i}')}[\hat{Q}_{\phi}^{\pi}(\mathbf{s}_{i}',\mathbf{a}')-\log \pi_{\theta}(\mathbf{a}'\mid \mathbf{s}_{i}')]
$$
Also, it's very common to add a "temperature" $\beta$ as a coefficient to the entropy terms.
## Inverse Reinforcement Learning
In standard imitation learning, e.g. behavioral cloning, the model attempts to copy the *actions* the expert takes, without any reasoning about the action outcomes. In contrast, humans, when learning, attempt to copy the *intent* of the expert, and consequently may take very different actions due to their reasoning about the expert's intent. Thus, it's potentially very useful to <font color="#c0504d">learn reward functions from demonstrations</font>, rather than have the developer define a reward function for the model to optimize.

Our previous soft optimality or control as inference framework is very helpful for this. Let's use the same probabilistic model with the same optimality variable; but now, instead of learning a policy based on a reward function, we will attempt to learn a reward function given trajectories sampled from a *soft-optimal* policy. In other words, we have
$$
p(\mathcal{O}_{t}\mid \mathbf{s}_{t},\mathbf{a}_{t},\psi)=\exp(r_{\psi}(\mathbf{s}_{t},\mathbf{a}_{t}))
$$
and consequently
$$
p(\tau \mid \mathcal{O}_{1:T},\psi) \propto p(\tau)\exp\left( \sum_{t}r_{\psi}(\mathbf{s}_{t},\mathbf{a}_{t}) \right)
$$
Note that there's a hidden denominator involved with the expression as well that's not shown in the proportionality equation. In particular, the denominator is $Z$, the partition function $Z=\int p(\tau)\exp(r_{\psi}(\tau)) \,\mathrm{d}\tau$.

Maximum likelihood learning tells us that the optimal $\psi$ is simply
$$
\hat{\psi}=\underset{\psi}{\arg\max}\ \frac{1}{N}\sum_{i=1}^{N} \log p(\tau_{i}\mid \mathcal{O}_{1:T},\psi)
$$
Also, we may note that, when taking the gradient of $p(\tau \mid \mathcal{O}_{1:T},\psi)$ w.r.t $\psi$, $p(\tau)$ is irrelevant since it represents the environment dynamics and is thus independent of the reward parameters. Therefore, we're really considering
$$
\mathcal{L}=\frac{1}{N}\sum_{i=1}^{N} r_{\psi}(\tau_{i})-\underbrace{ \log Z }_{ \text{normalizer} }
$$
where $\log Z$ shows up due to the hidden $Z$ denominator, as aforementioned.
$$
\begin{align*}
\nabla_{\psi}\mathcal{L} &= \frac{1}{N}\sum_{i=1}^{N} \nabla_{\psi}r_{\psi}(\tau_{i})-\underbrace{ \frac{1}{Z}\int p(\tau)\exp(r_{\psi}(\tau)) }_{ p(\tau \mid \mathcal{O_{1:T},\psi}) }\nabla_{\psi}r_{\psi}(\tau) \,\mathrm{d}\tau \\
&= \mathbb{E}_{\tau \sim \pi^{*}(\tau)}[\nabla_{\psi}r_{\psi}(\tau_{i})]-\mathbb{E}_{\tau \sim p(\tau \mid \mathcal{O}_{1:T},\psi)}[\nabla_{\psi}r_{\psi}(\tau)]
\end{align*}
$$
Notably, the gradient is zero when $\pi^{*}(\tau)$, the policy of our expert that we estimate with samples, is equivalent to $p(\tau \mid \mathcal{O}_{1:T},\psi)$, the soft optimal policy produced by following our current estimated reward function.

Additionally, because the partition function $Z$ integrates over all possible trajectories, it is of course intractable to compute directly. Therefore, we must estimate it instead—we can use variational inference, which dictates that we should train a maximum entropy RL agent to approximate the intractable optimal policy for the current reward function $r_{\psi}$. Only then may we sample from this policy to produce trajectories $\tau$ for the second term $\mathbb{E}_{\tau \sim p(\tau \mid \mathcal{O}_{1:T},\psi)}[\nabla_{\psi}r_{\psi}(\tau)]$.
<div style="page-break-after: always;"></div>

