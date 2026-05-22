# Lecture 5: Policy Gradients
## REINFORCE: Basic Policy Gradients
Recall the objective of RL
$$
\theta^{*} = \underset{\theta}{\arg\max}\ \mathbb{E}_{\tau\sim p_{\theta}(\tau)}\left[ \sum_{t} r(\mathbf{s}_{t},\mathbf{a}_{t}) \right]
$$
with sequences $\tau$ sampled from the trajectory distribution
$$
p_{\theta}(\mathbf{s}_{1},\mathbf{a}_{1},\dots ,\mathbf{s}_{H},\mathbf{a}_{H}) = p(\mathbf{s}_{1})\prod_{t=1}^{H} \pi_{\theta}(\mathbf{a}_{t}\mid \mathbf{s}_{t})p(\mathbf{s}_{t+1}\mid \mathbf{s}_{t},\mathbf{a}_{t})
$$
With policy gradient methods, we evaluate the policy by computing an estimate of our average reward, i.e.
$$
J(\theta) = \ \mathbb{E}_{\tau\sim p_{\theta}(\tau)}\left[ \sum_{t} r(\mathbf{s}_{t},\mathbf{a}_{t}) \right]\ \approx \frac{1}{N}\sum_{i}\sum_{t}r(\mathbf{s}_{t}^{(i)},\mathbf{a}_{t}^{(i)})
$$
where $J(\theta)$ is the value being maximized in the RL objective,
$$
\theta^{*} = \underset{\theta}{\arg\max}\underbracket[1][1]{\ \mathbb{E}_{\tau\sim p_{\theta}(\tau)}\left[ \sum_{t} r(\mathbf{s}_{t},\mathbf{a}_{t}) \right]\ }_{J(\theta)}
$$
and improve it by computing a gradient on the parameters
$$
\theta\leftarrow\theta+\alpha \nabla_{\theta}J(\theta)
$$
One might think that we're done with just this—we can go train a neural network now and compute gradients $\nabla_{\theta}J(\theta)$ to learn the correct policy. However, whatever machine learning library we're using doesn't know about the *external environment* that produces the feedback reward according to our policy's decisions. Therefore, we'll just receive a gradient of $0$, since our program doesn't even know the variables are influenced by the policy $\pi_{\theta}$; this is evidenced most clearly by the fact that our estimate of $J(\theta)$, i.e. $\frac{1}{N}\sum_{i}\sum_{t}r(\mathbf{s}_{t}^{(i)},\mathbf{a}_{t}^{(i)})$, doesn't even include $\theta$ in the expression!
### Direct Policy Differentiation
So here's how **direct policy differentiation** is done to optimize with respect to $\theta$ instead of the usual methods.

Let $r(\tau)=\sum_{t=1}^Hr(\mathbf{s}_{t},\mathbf{a}_{t})$. Then, we can write that
$$
J(\theta) = \mathbb{E}_{\tau \sim p_{\theta}(\tau)}[r(\tau)]=\int p_{\theta}(\tau)r(\tau) \,\mathrm{d}\tau
$$
Deriving with respect to $\theta$ produces
$$
\nabla_{\theta}J(\theta)=\int \nabla_{\theta}p_{\theta}(\tau)r(\tau) \,\mathrm{d}\tau 
$$
where the derivative operator may be moved inside the integral due to linearity of expectation. Now, note the following identity.

> [!info]
> $$
> p_{\theta}(\tau)\nabla_{\theta}\log p_{\theta}(\tau)=p_{\theta}(\tau) \frac{\nabla_{\theta}p_{\theta}(\tau)}{p_{\theta}(\tau)}= \nabla_{\theta}p(\tau)
> $$

Which allows us to substitute
$$
\nabla_{\theta}J(\theta)=\int p_{\theta}(\tau)\nabla_{\theta}\log p_{\theta}(\tau)r(\tau) \,\mathrm{d}\tau
$$
This is important because it allows us to rewrite this as an expectation!
$$
\nabla_{\theta}J(\theta)=\int (p_{\theta}(\tau))(\nabla_{\theta}\log p_{\theta}(\tau)r(\tau)) \,\mathrm{d}\tau = \mathbb{E}_{\tau \sim p_{\theta}(\tau)}[\nabla_{\theta}\log p_{\theta}(\tau)r(\tau)] 
$$
This conclusion is sometimes called the **policy gradient theorem**, and is very important—make sure you understand this!!

There's still a little bit of work to be done to finish this off, though. It's not immediately obvious how to compute $\log p_{\theta}(\tau)$. First, note that
$$
p_{\theta}(\tau)=p_{\theta}(\mathbf{s}_{1},\mathbf{a}_{1},\dots ,\mathbf{s}_{H},\mathbf{a}_{H}) = p(\mathbf{s}_{1})\prod_{t=1}^{H} \pi_{\theta}(\mathbf{a}_{t}\mid \mathbf{s}_{t})p(\mathbf{s}_{t+1}\mid \mathbf{s}_{t},\mathbf{a}_{t})
$$
Thus,
$$
\begin{align*}
\log p_{\theta}(\tau)&=\log p(\mathbf{s}_{1})+\sum_{t=1}^{H} \left[\log \pi_{\theta}(\mathbf{a}_{t}\mid \mathbf{s}_{t}) + \log p(\mathbf{s}_{t+1}\mid \mathbf{s}_{t},\mathbf{a}_{t})\right] \\
\nabla_{\theta}\log p_{\theta}(\tau) &= \cancel{ \log p(\mathbf{s}_{1}) }+\sum_{t=1}^{H} \left[\nabla_{\theta}\log \pi_{\theta}(\mathbf{a}_{t}\mid \mathbf{s}_{t}) + \log \cancel{ p(\mathbf{s}_{t+1}\mid \mathbf{s}_{t},\mathbf{a}_{t}) }\right] \\
&= \sum_{t=1}^{H} \nabla_{\theta}\log \pi_{\theta}(\mathbf{a}_{t}\mid \mathbf{s}_{t}) \\
\nabla_{\theta}J(\theta) &= \mathbb{E}_{\tau \sim p(\tau)}\left[ \left( \sum_{t=1}^{H} \nabla_{\theta}\log \pi_{\theta}(\mathbf{a}_{t}\mid \mathbf{s}_{t}) \right)\left( \sum_{t=1}^{H} r(\mathbf{s}_{t},\mathbf{a}_{t}) \right) \right]
\end{align*}
$$
Consequently, to evaluate the gradient, we can ***directly*** *compute it from the policy evaluations*!
$$
\nabla_{\theta}J(\theta) \approx \frac{1}{N}\sum_{i=1}^{N} \left[ \left( \sum_{t=1}^{H} \nabla_{\theta}\log \pi_{\theta}(\mathbf{a}_{t}^{(i)}\mid \mathbf{s}_{t}^{(i)}) \right)\left( \sum_{t=1}^{H} r(\mathbf{s}_{t}^{(i)},\mathbf{a}_{t}^{(i)}) \right) \right]
$$
This algorithm, in fact, is known as **REINFORCE**:

1. Sample $\tau$ (run policy)
2. Perform direct policy differentiation
3. Update $\theta\leftarrow\theta+\alpha \nabla_{\theta}J(\theta)$
### Maximum Likelihood Estimation
We can compare this policy gradient with the gradient for imitation learning computed by maximum likelihood estimation.
$$
\begin{align*}
\text{policy gradient:} && \nabla_{\theta}J(\theta) &\approx \frac{1}{N}\sum_{i=1}^{N} \left[ \left( \sum_{t=1}^{H} \nabla_{\theta}\log \pi_{\theta}(\mathbf{a}_{t}^{(i)}\mid \mathbf{s}_{t}^{(i)}) \right)\left( \sum_{t=1}^{H} r(\mathbf{s}_{t}^{(i)},\mathbf{a}_{t}^{(i)}) \right) \right] \\
\text{maximum likelihood:} && \nabla_{\theta}J_{\text{ML}}(\theta) &\approx \frac{1}{N}\sum_{i=1}^{N} \left( \sum_{t=1}^{H} \nabla_{\theta}\log \pi_{\theta}(\mathbf{a}_{t}^{(i)}\mid \mathbf{s}_{t}^{(i)}) \right)
\end{align*}
$$
So, the policy gradient is identical except for the *reward term*; thus, the policy gradient doesn't just try to imitate the sample trajectories, but it also favors positive rewards and penalizes negative rewards. Essentially, you can think of policy gradient as <font color="#c0504d">formalizing trial and error</font>!

> [!question] Partial Observability?
> Does policy gradient still function for RL problems with partial observability, i.e $\mathbf{o}_{t}\neq \mathbf{s}_{t}$? The answer is yes—it functions pretty much identically, just with $\log \pi_{\theta}(\mathbf{a}_{t}^{(i)}\mid \mathbf{o}_{t}^{(i)})$ instead, because the Markov property was never used in the derivation.
## Limitations
Policy gradient methods, however, have some limitations. For example, consider applying REINFORCE to an RL problem where a model plays from a preset chess position. The reward is $+1$ for winning and $-1$ for losing.

Naturally, the goal is for the model to favor good moves and penalize bad moves; for this to happen, we'd like $\nabla_{\theta}J(\theta)$ to produce positive multipliers for good moves and negative multipliers for bad moves. But this is not necessary the case, since other factors influence the model's winning chances!

- The quality of the starting position
- Incorrect multipliers when a good move is made, only to be followed later by a bad move (or vice versa)

Now, while these issues do average out, it takes *many, many samples*! The key problem is that this policy gradient algorithm has **high variance**. How can we reduce it?
## Variance Reduction
### Baselines
We can essentially *demean* the trajectories' rewards by computing
$$
b= \frac{1}{N}\sum_{i=1}^{N} r(\tau)
$$
and modifying $\nabla_{\theta}J(\theta)$ to
$$
\nabla_{\theta}J(\theta) \approx \frac{1}{N}\sum_{i=1}^{N} \nabla_{\theta}\log p_{\theta}(\tau)[r(\tau)-b] = \frac{1}{N}\sum_{i=1}^{N} [\nabla_{\theta}\log p_{\theta}(\tau)r(\tau)-\nabla_{\theta}\log p_{\theta}(\tau)b]
$$
Notably, the estimator of $\nabla_{\theta}J(\theta)$ remains *unbiased in expectation*!
$$
\begin{align*}
\mathbb{E}[\nabla_{\theta}\log p_{\theta}(\tau)b] &= \int p_{\theta}(\tau)\nabla_{\theta}\log p_{\theta}(\tau )b \,\mathrm{d}\tau  \\
&= \int \nabla_{\theta}p_{\theta}(\tau)b \, \mathrm{d}\tau && (1) \\
&= b\nabla_{\theta}\int p_{\theta}(\tau) \,\mathrm{d}\tau && (2) \\
&= b\nabla_{\theta}(1) && (3) \\
&= 0
\end{align*}
$$
Where $(1)$ derives from the identity we showed earlier, $(2)$ derives from linearity of expectation, and $(3)$ derives from the properties of a probability distribution, i.e. $\int p(x) \,\mathrm{d}x = 1$ for any probability distribution $p$ over $x$.

However, for some values of $b$, it will *reduce variance*! Letting $b$ be the average reward (i.e. defined above) does reduce variance, and generally performs well (though there exists better, but the optimal baseline is not used in practice due to its complexity).

The intuition here is that, even if all the trajectories' rewards are positive, we still induce the model to more strongly favor the higher rewards, rather than favor all positive rewards.
### Causality
Causality means that *a policy at time $t'$ cannot affect reward at time $t$ when $t<t'$*. At the moment, our gradient calculation does not consider causality.
$$
\nabla_{\theta}J(\theta) \approx \frac{1}{N}\sum_{i=1}^{N} \left[ \left( \sum_{t=1}^{H} \nabla_{\theta}\log \pi_{\theta}(\mathbf{a}_{t}^{(i)}\mid \mathbf{s}_{t}^{(i)}) \right)\left( \sum_{t=1}^{H} r(\mathbf{s}_{t}^{(i)},\mathbf{a}_{t}^{(i)}) - b_{t} \right) \right]
$$
The gradient $\nabla_{\theta}\log \pi_{\theta}(\mathbf{a}_{t}^{(i)}\mid \mathbf{s}_{t}^{(i)})$ is affected by rewards across *all* time steps. Instead, by including causality, we produce
$$
\nabla_{\theta}J(\theta) \approx \frac{1}{N}\sum_{i=1}^{N} \sum_{t=1}^{H} \nabla_{\theta}\log \pi_{\theta}(\mathbf{a}_{t}^{(i)}\mid \mathbf{s}_{t}^{(i)}) \left( \sum_{t'=t}^{H} r(\mathbf{s}_{t'}^{(i)},\mathbf{a}_{t'}^{(i)}) - b_{t'} \right)
$$
So that the gradient at time step $t$ is only influenced by rewards from time steps $t'\in[t,H]$.

Note that this reduces variance for a rather trivial reason actually—we're multiplying $\nabla_{\theta}\log \pi_{\theta}$ by smaller numbers. Regardless, this modification is typically effective.
## Practical Implementation
### Automatic Differentiation
We can compute the policy gradients $\nabla_{\theta}\log \pi_{\theta}(\mathbf{a}_{t}^{(i)}\mid \mathbf{s}_{t}^{(i)})$ with **automatic differentiation**, actually. They key idea is that these gradients are the same as the gradients for a supervised neural network; we just have to design the right graph such that its overall gradient is the policy gradient $\nabla_{\theta}J(\theta)$.

We have to essentially set the loss function to be a *weighted maximum likelihood* loss function (recall the [[#Maximum Likelihood Estimation|comparison]] between maximum likelihood and policy gradient), weighted by reward.
$$
\tilde{J}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{H} \log \pi_{\theta}(\mathbf{a}_{t}^{(i)}\mid \mathbf{s}_{t}^{(i)})\hat{Q}_{t}^{(i)}
$$
where $\hat{Q}_{t}^{(i)}=\sum_{t=1}^{H}r(\mathbf{s}_{i}^{(t)},\mathbf{a}_{t}^{(i)})$. $\log \pi_{\theta}$ is cross-entropy for discrete problems and MSE for continuous (Gaussian) problems, like with maximum likelihood.

The pseudocode for maximum likelihood (discrete) for supervised learning would look like
```python
logits = policy.predictions(states)
negative_likelihoods = tf.nn.softmax_cross_entropy_with_logits(labels=actions, logits=logits)
loss = tf.reduce_mean(negative_likelihoods)
gradients = loss.gradients(loss, variables)
```
While the pseudocode for policy gradient (discrete) for reinforcement learning would look like
```python
logits = policy.predictions(states)
negative_likelihoods = tf.nn.softmax_cross_entropy_with_logits(labels=actions, logits=logits)
weighted_negative_likelihoods = tf.multiply(negative_likelihoods, q_values)
loss = tf.reduce_mean(weighted_negative_likelihoods)
gradients = loss.gradients(loss, variables)
```
And that's it!
### Reminders
- High variance gradients
- Larger batch sizes may be better
- Tweaking learning rates is hard!
<div style="page-break-after: always;"></div>

