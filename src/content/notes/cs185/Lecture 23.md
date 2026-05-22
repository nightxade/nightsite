# Lecture 23: Exploration and Skill Learning
What if want an RL agent to recover diverse behavior *without* any reward function? In other words, the problem is inducing an agent to prepare for an **unknown** future goal.
## Information Theory Recap
### Some Useful Identities
$$
\begin{align*}
\mathcal{H}(p(\mathbf{x})) &= -\mathbb{E}_{\mathbf{x}\sim p(\mathbf{x})}[\log p(\mathbf{x})] \\
\mathcal{I}(\mathbf{x};\mathbf{y}) &= D_{\text{KL}}(p(\mathbf{x},\mathbf{y})\parallel p(\mathbf{x})p(\mathbf{y})) \\
&= \mathcal{H}(p(\mathbf{y}))-\mathcal{H}(p(\mathbf{y}\mid \mathbf{x}))
\end{align*}
$$
### RL Information Theoretic Quantities
For RL exploration, we are concerned about a few values from information theory. In particular, we may be concerned with the state marginal entropy of policy $\pi$, i.e. $\mathcal{H}(\pi(\mathbf{s}))$. We are also especially interested in a quantity known as **empowerment**,
$$
\mathcal{I}(\mathbf{a}_{t}\mid \mathbf{s}_{t+1})=\mathcal{H}(\mathbf{s}_{t+1})-\mathcal{H}(\mathbf{s}_{t+1}\mid \mathbf{a}_{t})
$$
which essentially measures the influence of the action $\mathbf{a}_{t}$ on determining the next state $\mathbf{s}_{t+1}$.
## Learning Skills with Empowerment
We'd like to learn some policy $\pi(\mathbf{a}\mid \mathbf{s},z)$ where it depends on the specific skill $z$, i.e. $z$ can be treated as an index into a set of skills. Each skill should do something different—visit different **state space regions**—such that they sufficiently cover the state space.

As one might expect, we can treat this as a multitask RL problem, where each skill has its own reward function.
$$
\pi(\mathbf{a}\mid \mathbf{s})=\underset{\pi}{\arg\max}\ \sum_{z}\mathbb{E}_{\mathbf{s}\sim \pi(\mathbf{s}\mid z)}[r(\mathbf{s},z)]
$$
However, as aforementioned, we do not know a priori what sort of tasks we want to learn, and therefore what the reward function for each skill should look like. Instead, we use a reward function that *changes over the course of training*. Essentially, each skill's reward function will encourage it to visit states that are *unlikely for other skills* $z'\neq z$. We do this by defining the reward function as
$$
r(s,z)=\log p(z\mid \mathbf{s})
$$
where $p(z\mid \mathbf{s})$ is a **discriminator** model trained to predict the skill being executed given an environment state. Notably, there is a connection between optimizing this objective and mutual information
$$
\mathcal{I}(z;\mathbf{s}) = \mathcal{H}(z)-H(z\mid \mathbf{s})
$$
The first term is determined by the prior $p(z)$ (i.e. maximized by uniform prior over $p(z)$), while the second term is affected by $p(z\mid \mathbf{s})$; in particular, the entropy of $z\mid \mathbf{s}$ is minimized when it is easy to predict the skill from a given state. In other words, the above objective maximized $\mathcal{I}(z;\mathbf{s})$, the mutual information between the skills and states.
## Reaching Goals with Empowerment
These skills, while promoting diverse trajectories, may not really help with realizing RL goals. What if, instead of conditioning on an arbitrary skill index $z$, we conditioned on a **goal state** $\mathbf{g}$ that the agent should reach?
$$
\pi(\mathbf{a}\mid \mathbf{s},\mathbf{g})=\underset{\pi}{\arg\max}\ \sum_{\mathbf{g}}\mathbb{E}_{\mathbf{s}\sim \pi(\mathbf{s}\mid \mathbf{g})}[r(\mathbf{s},\mathbf{g})]
$$
For now, we'll keep the same reward function design.
$$
r(\mathbf{s},\mathbf{g})=\log p(\mathbf{g}\mid \mathbf{s})
$$
If we consider the same mutual information equation
$$
\mathcal{I}(\mathbf{g};\mathbf{s}) = \mathcal{H}(\mathbf{g})-\mathcal{H}(\mathbf{g}\mid \mathbf{s})
$$
The second term is again minimized by maximizing $\log p(\mathbf{g}\mid \mathbf{s})$; however, the first term is a bit less clear. We don't necessarily know all the valid states beforehand—for instance, consider an RL task training on image states. If we just maximize entropy over random images, practically all of these aren't going to be valid states. This is, notably, less of an RL problem and more of a general ML problem.

> [!info]
> In practice, other rewards are used, e.g. $r(\mathbf{s},\mathbf{g})=\delta(\mathbf{s}=\mathbf{g})$ or $r(\mathbf{s},\mathbf{g})=\delta(\lVert \mathbf{s}-\mathbf{g} \rVert\leq\epsilon)$.
### Example: Skew-Fit
Skew-Fit is an old algorithm (Skew-Fit: State-Covering Self-Supervised Reinforcement Learning, Dalal et al.) that is inspired by such ideas.

1. Propose goal $\mathbf{g}\sim p_{\psi}(\mathbf{g})$.
2. Attempt to reach goal using $\pi(\mathbf{a}\mid \mathbf{s},\mathbf{g})$.
3. Use data to update $\pi$.
4. Use data to update $p_{\psi}(\mathbf{g})$.

This procedure doesn't quite work, unfortunately, because the replay buffer is used to train $p_{\psi}(\mathbf{g})$, and this method directly exacerbates the imbalances/bias in the training distribution dataset. For instance, if there are two possible goals that are additionally orthogonal, and the training distribution favors one goal in the training set, the RL agent will be trained to be biased towards taking actions for this goal.

Thus, we need to somehow rebalance the training set to reduce its bias. The trick is to use a **weighted** MLE estimator instead of the standard MLE estimator
$$
\psi\leftarrow \underset{\psi}{\arg\max}\ \mathbb{E}[w(\mathbf{g})\log p(\mathbf{g})]
$$
where $w(\mathbf{g})=p_{\psi}(\mathbf{g})^{\alpha}$ for $\alpha \in[-1,0)$. This increases the entropy $\mathcal{H}(p_{\psi}(\mathbf{g}))$. Note that this does not erroneously encourage invalid states, since it only increases the weight of states from the training set, which are valid.
<div style="page-break-after: always;"></div>
<div style="page-break-after: always;"></div>

