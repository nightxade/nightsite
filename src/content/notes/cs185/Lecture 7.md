# Lecture 7: Value-Based RL
## Actor-Critic without the Actor
Recall the off-policy actor-critic algorithm from [[Lecture 6#Off-Policy Actor-Critic|the previous lecture]]. Now, image we have a *small, discrete* action space. Instead of having a distinct actor model and critic model, we can just have a critic model, and define the actor's policy implicitly as based on the critic, i.e.
$$
\pi_{\theta}(\mathbf{a}\mid \mathbf{s}) = \left\{ \begin{matrix}
1, & \mathbf{a}=\underset{\mathbf{a}}{\arg\max}\ \hat{Q}_{\phi}^{\pi}(\mathbf{s},\mathbf{a}) \\
0, & \text{otherwise}
\end{matrix} \right.
$$
This simplifies our algorithm to the following steps. Note that we'll rename our parameters to $Q_{\theta}$ instead of $Q_{\phi}$ to follow standard terminology.

1. Get $(\mathbf{s}_{i},\mathbf{a}_{i},\mathbf{s}_{i}')$ by taking one step, store in replay buffer
2. Evaluate $y_{i}=r(\mathbf{s}_{i},\mathbf{a}_{i})+\gamma \mathbb{E}_{\mathbf{a}'\sim \pi_{\theta}(\mathbf{a}'\mid \mathbf{s}'_{i})}[\hat{Q}_{\theta}(\mathbf{s}_{i}',\mathbf{a}')]=r(\mathbf{s}_{i},\mathbf{a}_{i})+\gamma \underset{\mathbf{a}'}{\max}\ \hat{Q}_{\theta}(\mathbf{s}_{i}',\mathbf{a}')$.
3. Update $\phi$ using $\nabla_{\phi}\sum_{i=1}^{B}\lVert \hat{Q}_{\theta}(\mathbf{s}_{i},\mathbf{a}_{i})-y_{i} \rVert^{2}$.

This is known as an *off-policy critic* algorithm, and is known as **$Q$-learning**.
## Policy Iteration and Dynamic Programming
We'll now discuss a different sort of thinking path that leads to the same conclusion, i.e. $Q$-learning.

**Policy iteration**, at a high level, is a two-step algorithm.

1. **Policy Evaluation:** Fit $Q^{\pi}$ or $V^{\pi}$.
2. **Policy Improvement**: Improve the policy $\pi\leftarrow \pi'$.

Actor-critic and $Q$-learning are both algorithms within this family.
### Value Function DP
First, we assume knowledge of $p(\mathbf{s}'\mid \mathbf{s},\mathbf{a})$ and that the state and action space are small and discrete. For instance, consider a small grid of states, with an action space of moving up, down, left, or right. We can store the entirety of $V^{\pi}(\mathbf{s})$ in a table, and perform *bootstrapped* policy evaluation
$$
V^{\pi}(\mathbf{s})\leftarrow \mathbb{E}_{\mathbf{a}\sim \pi(\mathbf{a}\mid \mathbf{s})}[r(\mathbf{s},\mathbf{a})+\gamma \mathbb{E}_{\mathbf{s}'\sim p(\mathbf{s}'\mid \mathbf{s},\mathbf{a})}[V^{\pi}(\mathbf{s}')]]
$$
where we calculate over the entire space, i.e. all $(\mathbf{s}',\mathbf{a}')$, which provides an exact calculation, not an estimate (no sampling), of the updated value function. Afterwards, we may update our policy, i.e.
$$
\pi_{\theta}(\mathbf{a}\mid \mathbf{s}) = \left\{ \begin{matrix}
1, & \mathbf{a}=\underset{\mathbf{a}}{\arg\max}\ \hat{Q}_{\phi}^{\pi}(\mathbf{s},\mathbf{a}) \\
0, & \text{otherwise}
\end{matrix} \right.
$$
to produce a **deterministic** policy. Notably, the deterministic policy actually allows us to simplify the policy evaluation equation to
$$
V^{\pi}(\mathbf{s})\leftarrow r(\mathbf{s},\pi(\mathbf{s}))+\gamma \mathbb{E}_{\mathbf{s}'\sim p(\mathbf{s}'\mid \mathbf{s},\mathbf{a})}[V^{\pi}(\mathbf{s}')]
$$
Notably, since there is *no sampling*, and thus no policy running, this is simply dynamic programming, and not really any reinforcement learning! This is also known as **tabular policy learning**.
### Value Iteration
We can also use $Q$ functions for DP. In essence, we change our policy evaluation step to
$$
Q^{\pi}(\mathbf{s},\mathbf{a})\leftarrow r(\mathbf{s},\mathbf{a})+\gamma \mathbb{E}_{\mathbf{s}'\sim p(\mathbf{s}'\mid \mathbf{s},\mathbf{a})}[V^{\pi}(\mathbf{s}')]
$$
and change our policy improvement step to
$$
V^{\pi}(\mathbf{s})\leftarrow \underset{\mathbf{a}}{\max}\ Q^{\pi}(\mathbf{s},\mathbf{a})
$$
where our policy is implicitly defined by *greedily* following the value function. This is known as **value iteration**, which is essentially just a one step algorithm.
## Fitted Value Iteration
Now, let's imagine our state space has grown much larger, and thus it is no longer tractable to represent $V^{\pi}$ in a tabular format. How do we represent $V(\mathbf{s})$?

Of course, the natural idea is to use a neural network that regresses on $V(\mathbf{s})$ with an MSE loss function. This produces the **fitted value iteration** algorithm.

1. Set $y_{i}\leftarrow \underset{\mathbf{a}_{i}}{\max}\ (r(\mathbf{s}_{i},\mathbf{a}_{i})+\gamma \mathbb{E}[V_{\theta}(\mathbf{s}_{i}')])$.
2. Set $\theta\leftarrow \underset{\theta}{\arg\min}\ \frac{1}{2}\sum_{i}\lVert V_{\theta}(\mathbf{s}_{i})-y_{i} \rVert^{2}$.

Notably, we still require knowledge of the state transition probabilities and the action space must be small and discrete, due to the nature of step 1.

In practice, however, we won't know state transition probabilities, and we need to use samples rather than iterating over the entire action space...
### Fitted $Q$ Iteration
The trick is to simply use $Q$-functions instead of value functions!

1. Set $y_{i}\leftarrow r(\mathbf{s}_{i},\mathbf{a}_{i})+\gamma \mathbb{E}[V_{\theta}(\mathbf{s}')]$, where we can approximate $\mathbb{E}[V_{\theta}(\mathbf{s}')]\approx \underset{\mathbf{a}'}{\max}\ Q_{\theta}(\mathbf{s}_{i}',\mathbf{a}_{i}')$.
2. Set $\theta\leftarrow \underset{\theta}{\arg\min}\ \frac{1}{2}\sum_{i}\lVert Q_{\theta}(\mathbf{s}_{i},\mathbf{a}_{i})-y_{i} \rVert^{2}$.

The key idea is that the $Q$ function, $Q:\mathcal{S}\times \mathcal{A}\to \mathbb{R}$, is also used to compute an expected value over actions, but does so without needing extra sampling/knowing the state transition probabilities; the best action is already known according to the estimated $Q_{\theta}$, by taking the max value over existing samples.

> [!question]- Why doesn't this work with value functions?
> With value functions, if we were to compute $y_{i}$ using samples $\mathbf{a}_{i}$, we would be evaluating our current policy $\pi'$ using samples from an old policy $\pi$... not good! This results in an outdated policy evaluation, and therefore the target values for $V_{\theta}$ aren't representative of the true, current $V_{\theta}$, i.e. the labels $y_{i}$ are outdated. Hence, fitted value iteration is *not* an off-policy algorithm; it does not use any samples, but rather iterates over the entire action space and uses its knowledge of the state transition probabilities to compute its estimate of the value function. Fitted $Q$ iteration *is* able to act as an off-policy algorithm, and uses samples to estimate $Q$.

> [!tip]-
> There are two ways to model the $\underset{\mathbf{a}'}{\arg\max}\ Q_{\theta}(\mathbf{s}_{i}',\mathbf{a}_{i}')$.
> 
> 1. Make a neural network that takes in $\mathbf{s}_{i},\mathbf{a}_{i}$ and outputs the expected $Q$-value. This is the intuitive method (model $Q(\mathbf{s},\mathbf{a})$ directly), and is the implied method above. (Max taken over existing samples).
> 2. If the action space is small, make a neural network that takes in $\mathbf{s}_{i}$ and outputs the expected $Q$-value for all $\mathbf{a}_{i}$. Then, the arg max is just computed directly over the resulting outputs, without needing to use existing sample actions.

$Q$-learning works for off-policy samples, and only has one neural network, and therefore no high-variance policy gradient. However, it lacks convergence guarantees for nonlinear function approximations—an issue we'll discuss more later.

Our full fitted $Q$-iteration algorithm is thus

1. Collect $B$ samples $(\mathbf{s}_{i},\mathbf{a}_{i},\mathbf{s}_{i}')$ to add to the replay buffer.
2. Set $y_{i}=r(\mathbf{s}_{i},\mathbf{a}_{i})+\gamma \underset{\mathbf{a}'}{\max}\ Q_{\theta}(\mathbf{s}_{i}',\mathbf{a}_{i}')$.
3. Set $\theta\leftarrow \underset{\theta}{\arg\min}\ \frac{1}{2}\sum_{i}\lVert Q_{\theta}(\mathbf{s}_{i},\mathbf{a}_{i})-y_{i} \rVert^{2}$.
4. Repeat steps 2-3 $K$ times, for some hyperparameter $K$.
5. Go back to step 1 and repeat.
## Return to $Q$-Learning
### $Q$-Iteration Optimization Objective
First, what is fitted $Q$-iteration really optimizing?

Well, it's performing some regression, and attempting to compute parameters $\theta$ for our model $Q_{\theta}$ that minimize the error $\mathcal{E}$
$$
\begin{align*}
\mathcal{E}&=\frac{1}{2}\mathbb{E}_{(\mathbf{s},\mathbf{a})\sim\beta}[(Q_{\theta}(\mathbf{s},\mathbf{a})-y_{i})^{2}] \\
&=\frac{1}{2}\mathbb{E}_{(\mathbf{s},\mathbf{a})\sim\beta}[(Q_{\theta}(\mathbf{s},\mathbf{a})-[r(\mathbf{s},\mathbf{a})+\gamma \underset{\mathbf{a}'}{\max}\  Q_{\theta}(\mathbf{s}',\mathbf{a}')])^{2}]
\end{align*}
$$
where $\beta$ is the dataset/replay buffer we're sampling from.

If $\mathcal{E}=0$, then $Q_{\theta}(\mathbf{s},\mathbf{a})=r(\mathbf{s},\mathbf{a})+\gamma \underset{\mathbf{a}'}{\max}\ Q_{\theta}(\mathbf{s}',\mathbf{a}')$, and $Q_{\theta}$ is an **optimal** $Q$-function $Q^{*}$ that corresponds to the optimal policy $\pi^{*}$.
### Online $Q$-Learning
We can produce the **online $Q$-learning** algorithm by simply using fitted $Q$-iteration with $B,K=1$.

1. Take some action $\mathbf{a}$, observe $(\mathbf{s}_{i},\mathbf{a}_{i},\mathbf{s}_{i}')$.
2. $y_{i}=r(\mathbf{s}_{i},\mathbf{a}_{i})+\gamma \underset{\mathbf{a}'}{\max}\ Q_{\theta}(\mathbf{s}_{i}',\mathbf{a}_{i}')$.
3. $\theta\leftarrow\theta-\alpha \frac{\mathrm{d} Q_{\theta} }{\mathrm{d} \theta }(\mathbf{s}_{i},\mathbf{a}_{i})(Q_{\theta}(\mathbf{s}_{i},\mathbf{a}_{i})-y_{i})$.

Notably, this is still *off-policy*, unlike the [[Lecture 6#Online Actor-Critic|online actor-critic algorithm]] we covered last lecture. Thus, we have a choice in sampling $(\mathbf{s}_{i},\mathbf{a}_{i},\mathbf{s}_{i}')$; most importantly, it doesn't have to be the result of taking the best action according to our current policy. This allows us to ***explore*** more of the action space.
### Exploratory $Q$-Learning
We can extend this idea of *exploration* with **$\epsilon$-greedy methods**! In essence, we act according to the policy
$$
\pi(\mathbf{a}_{t}\mid \mathbf{s}_{t}) = \left\{ \begin{matrix}
1-\epsilon, & \text{if }\mathbf{a}_{t}=\underset{\mathbf{a}_{t}}{\arg\max}\ Q_{\theta}(\mathbf{s}_{t},\mathbf{a}_{t}) \\
\epsilon/(\lvert \mathcal{A} \rvert -1), & \text{otherwise}
\end{matrix}\right.
$$
which chooses the greedy action with probability $1-\epsilon$ and chooses a different action uniformly at random with probability $\epsilon$.

An alternative is to use a **soft max**, which ensures
$$
\pi(\mathbf{a}_{t}\mid \mathbf{s}_{t})\propto \exp(Q_{\theta}(\mathbf{s}_{t},\mathbf{a}_{t}))
$$
This is known as **Boltzmann exploration**, and basically weights the non-greedy actions according to their estimated values, rather than just choosing from them randomly.

> [!warning]
> Simply applying the $Q$-Learning algorithm as it is now is not sufficient; we need some tricks to make it work in practice, which we'll discuss in future lectures!
<div style="page-break-after: always;"></div>

