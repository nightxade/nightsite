# Lecture 6: Actor-Critic Algorithms
## Improving Policy Gradient with Values
Previously, we discussed the application of **causality** to reduce variance in policy gradient methods. In essence, we multiply the gradient for each time step $t$ by the "reward-to-go" $\hat{Q}_{t}^{(i)}$, the rewards received starting from this time step until the end of the sample.

This reward-to-go is actually a random variable, since the rest of the trajectory is random (since the policy itself is stochastic). (As in, if you were to start from the same state and action pair multiple times, the reward to go would vary across the iterations due to the stochasticity of the policy). Naturally, this random variable has some associated variance; in fact, we can produce an estimate of $\hat{Q}_{t}^{(i)}$ that has *lower variance* relative to our existing estimate, i.e.
$$
\hat{Q}_{t}^{(i)} \approx \sum_{t'=t}^{H} \mathbb{E}_{\pi_{\theta}}[r(\mathbf{s}_{t'},\mathbf{a}_{t'}\mid \mathbf{s}_{t}^{(i)},\mathbf{a}_{t}^{(i)})]
$$
Ideally, we'd like our estimator to be an unbiased estimate of the *true* expected reward-to-go.
$$
Q(\mathbf{s}_{t},\mathbf{a}_{t})=\sum_{t'=t}^{H} \mathbb{E}_{\pi_{\theta}}[r(\mathbf{s}_{t'},\mathbf{a}_{t'})\mid \mathbf{s}_{t},\mathbf{a}_{t}]
$$
The current $\hat{Q}_{t}^{(i)}$ is unbiased (it is the same formula, after all, just taken over a few trajectories instead of the whole space) but experiences high variance.
### Baseline?
Recall the **baseline** that replaced our reward-to-go estimate with $r(\tau)-b_{t}$, for some value $b_{t}$. What should this value be? Previously, we discussed how the *average reward* served as a good baseline. However, this was for policy gradient methods *without causality*—what's a good baseline for our reward-to-go, i.e. $\hat{Q}_{t}^{(i)}(\mathbf{s}_{t}^{(i)},\mathbf{a}_{t}^{(i)})=Q_{t}(\mathbf{s}_{t}^{(i)},\mathbf{a}_{t}^{(i)})-b_{t}$?

As it turns out, using the mean as a metric is still good; but, instead of averaging across rewards, we average across the possible $Q$-values for the state at time $t$ instead, or just the average reward-to-go.
$$
V(\mathbf{s}_{t})=\mathbb{E}_{\mathbf{a}_{t}\sim \pi_{\theta}(\mathbf{a}_{t}\mid \mathbf{s}_{t})}[Q(\mathbf{s}_{t},\mathbf{a}_{t})]
$$
This is known as the **value function**. As with the average reward baseline, this results in an estimator that retains unbiasedness while lowering variance. Thus,
$$
\nabla_{\theta}J(\theta) = \frac{1}{N}\sum_{i=1}^{N} \sum_{t=1}^{H} \nabla_{\theta}\log_{\theta}(\mathbf{a}_{t}^{(i)}\mid \mathbf{s}_{t}^{(i)})(Q(\mathbf{s}_{t}^{(i)},\mathbf{a}_{t}^{(i)})-V(\mathbf{s}_{t}^{(i)}))
$$
The expression $Q-V$ is so important it's denoted the **advantage**.
$$
A(\mathbf{s}_{t},\mathbf{a}_{t})=Q(\mathbf{s}_{t},\mathbf{a}_{t})-V(\mathbf{s}_{t})
$$
Positive advantage is assigned to actions that are *better than average* for the current state, and negative advantage is assigned to those that are worse than average.
### State/State-Action Value Functions
In short,

- $Q^{\pi}(\mathbf{s}_{t},\mathbf{a}_{t})$, the $Q$-function, is the total reward from taking $\mathbf{a}_{t}$ in $\mathbf{s}_{t}$.
- $V^{\pi}(\mathbf{s}_{t})$, the value function, is the total reward from $\mathbf{s}_{t}$.
- $A^{\pi}(\mathbf{s}_{t},\mathbf{a}_{t})$, the advantage function, is the measure of how much better $\mathbf{a}_{t}$ is for $\mathbf{s}_{t}$.

An **actor-critic** method will contain two models: one to capture the policy (actor), and one to capture the advantage function (critic). The latter estimator is typically a $Q$-function or value function.
## Policy Evaluation
Policy evaluation is the process of using a policy $\pi$ to estimate $V^{\pi}$ or $Q^{\pi}$.

$Q^{\pi}$ has a nice, recursive expression.
$$
Q^{\pi}(\mathbf{s}_{t},\mathbf{a}_{t})=r(\mathbf{s}_{t},\mathbf{a}_{t})+\mathbb{E}_{\mathbf{s}_{t+1}\sim p(\mathbf{s}_{t+1}\mid \mathbf{s}_{t},\mathbf{a}_{t})}[V^{\pi}(\mathbf{s}_{t+1})]
$$
which we can estimate with a single sample as
$$
Q^{\pi}(\mathbf{s}_{t},\mathbf{a}_{t})\approx r(\mathbf{s}_{t},\mathbf{a}_{t})+V^{\pi}(\mathbf{s}_{t+1})
$$
This introduces a bit of variance due to replacing an expectation with a single sample, but not too much since it's only over a single step. This results in a nice expression for the advantage, though, that is expressed *only in terms of $V^{\pi}(\mathbf{s})$*.
$$
A^{\pi}(\mathbf{s}_{t},\mathbf{a}_{t})\approx r(\mathbf{s}_{t},\mathbf{a}_{t})+V^{\pi}(\mathbf{s}_{t+1})-V^{\pi}(\mathbf{s}_{t})
$$
Thus, we can just try to fit a model $\hat{V}^{\pi}(\mathbf{s})$ with parameters $\phi$ to estimate $V^{\pi}(\mathbf{s})$. (Note, this is possible with just $Q$-functions too, but this is originally how actor-critic methods were done).

> [!tip]
> This estimator of $A^{\pi}$ remains *unbiased* because the expectation was replaced with a single sample taken from the true data generating distribution.

> [!info]
> The RL objective is just $J(\theta)=\mathbb{E}_{\mathbf{s}_{1}\sim p(\mathbf{s}_{1})}[V^{\pi}(\mathbf{s}_{1})]$.

So how do we fit a model $\hat{V}^{\pi}(s)$?
### Monte Carlo Method
**Monte Carlo** policy evaluation is the most natural method, and just involves generating some samples and averaging together (what our policy gradient method already does!). One drawback, however, is that to produce multiple samples for a single state, it requires *resetting* the simulator/environment back to the exact same state, which isn't necessarily possible; however, we can still use the single sample estimator.

Critically, this single sample estimator can still be effective! The key idea is that many states observed from our data will be similar, and a good neural network/model should be able to learn this similarity and effectively "average" their resulting trajectories/reward-to-go's together. Thus, the model we train to predict $V^{\pi}(\mathbf{s})$ will actually provide *better estimates* than the single sample estimates themselves.

> [!info]
> Actual Monte Carlo policy estimation, i.e. resetting the environment and measuring multiple trajectories, is better, but the model-based method works well in practice and is effectively an *approximation* of true Monte Carlo estimation.

The model is typically some sort of supervised regression, e.g. MSE regression
$$
\mathcal{L}(\phi)=\frac{1}{2}\sum_{i=1}^{N} \sum_{t=1}^{H} \lVert \hat{V}_{\phi}^{\pi}(\mathbf{s}_{t}^{(i)})-y_{t}^{(i)} \rVert ^{2}
$$
where the target label $y_{t}^{(i)}$ is
$$
y_{t}^{(i)}=r(\mathbf{s}_{t}^{(i)},\mathbf{a}_{t}^{(i)})+V^{\pi}(\mathbf{s}_{t+1}^{(i)})
$$
and $V^{\pi}$ is the measured value function for that trajectory, i.e. a single sample estimate of the value function.
### Bootstrapping
Instead of using the single-sample estimate of $V^{\pi}$ to generate our target labels, however, we can use our *previously fitted* value function $\hat{V}_{\phi}^{\pi}(\mathbf{s}_{t+1}^{(i)})$ to estimate $V^{\pi}(\mathbf{s}_{t+1}^{(i)})$, i.e.
$$
y_{t}^{(i)}=r(\mathbf{s}_{t}^{(i)},\mathbf{a}_{t}^{(i)})+\hat{V}_{\phi}^{\pi}(\mathbf{s}_{t+1}^{(i)})
$$
This is known as **bootstrapping**.

> [!tip]
> $y_{t}^{(i)}$, as a target value, is treated as a *constant* when calculating the gradient with respect to $\phi$. This is usually denoted by the **stop-gradient** operator $[\dots]_{\times}$, e.g.
> $$
> \mathcal{L}(\phi)= \mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[ \sum_{t=1}^{H} \bigg(\hat{V}_{\phi}^{\pi}(\mathbf{s}_{t}^{(i)})-\Big[r(\mathbf{s}_{t}^{(i)},\mathbf{a}_{t}^{(i)})+\hat{V}_{\phi}^{\pi}(\mathbf{s}_{t+1}^{(i)})\Big]_{\times }\bigg)^{2} \right]
> $$
> This is used for bootstrapping because $\phi$ is a parameter in calculating $\hat{V}_{\phi}^{\pi}(\mathbf{s}_{t+1}^{(i)})$, but its gradient should not be included when calculating $\nabla_{\phi}\mathcal{L}(\phi)$ because the target value $y_{t}^{(i)}$ should be a constant, and should not be "trained" or "learned." (Wouldn't really make sense for the model to learn to adjust the target value it's trying to match, now would it?). 

> [!warning]
> Use of bootstrapping creates a *biased* estimator, due to an **imperfect critic** $\hat{V}_{\phi}^{\pi}$. Despite this, it performs much better than the standard Monte Carlo estimator because it substantially reduces variance.

> [!info]
> Bootstrapping is known as **temporal difference (TD) learning**, which we will discuss in further detail in future lectures.

This has some issues, though, for *infinite horizon* RL problems. Imagine an environment where every single action (regardless of state) is $1$. Regardless of what the neural network/model of $\hat{V}$ is initialized to, its predictions will grow without bound, as the cyclical nature of bootstrapping causes increases in $y_{t}^{(i)}$ to increase $\hat{V}$, which increase $y_{t}^{(i)}$, which increases $\hat{V}$, etc.
### Discount Factors
We can fix this, though, with **discounting**! Discounting represents the idea that sooner rewards are better than later rewards with the same value, and is mathematically
$$
y_{t}^{(i)}=r(\mathbf{s}_{t}^{(i)},\mathbf{a}_{t}^{(i)})+\gamma \hat{V}_{\phi}^{\pi}(\mathbf{s}_{t+1}^{(i)})
$$
for some constant **discount factor** $\gamma \in[0,1]$ (in practice, $\gamma \approx1$, e.g. $0.99$).

Notably, this changes the MDP. Every state now has a $1-\gamma$ probability of transitioning to a "death" or absorbing state, in which no more rewards may be received/there are no transitions out of the death state. Thus, intuitively, discounting represents a belief that there is a $1-\gamma$ probability that the horizon will end now, and therefore it's more desirable to receive rewards sooner than later.

Discounting also helps *reduce variance*. By deprioritizing rewards further in the future, we reduce variance because those rewards are more uncertain (higher variance) than more immediate rewards.

> [!tip]
> With discounting, we practically always use the reward-to-go formulation, i.e. causality. Why? Because this discounts rewards relative to the current time step. Without causality, it would just discount later rewards *not* relative to the current time step, but relative to $t=0$.
### Time-Varying vs. Time-Invariant
Oftentimes, we don't actually care the time step at which data is collected! Instead, we only care about the **transition** probabilities, i.e. we consider $(\mathbf{s}_{i},\mathbf{a}_{i},\mathbf{s}_{i}')$ instead of $(\mathbf{s}_{t},\mathbf{a}_{t},\mathbf{s}_{t+1})$. This just lets us change notation around, i.e. training data becomes
$$
\{ (\mathbf{s}_{t}^{(i)},r(\mathbf{s}_{t}^{(i)},\mathbf{a}_{t}^{(i)})+\hat{V}_{\phi}^{\pi}(\mathbf{s}_{t+1}^{(i)})) \} \longrightarrow \{ (\mathbf{s}_{i},r(\mathbf{s}_{i},\mathbf{a}_{i})+\hat{V}_{\phi }^{\pi}(\mathbf{s}_{i}')) \}
$$
and similarly for other expressions. The problem itself doesn't change.
### Examples
- TD-Gammon played Backgammon, using a value function that just estimated the expected outcome given board state
- AlphaGo played Go, using the same value function but a bigger, more advanced model
## The Actor-Critic Algorithm
### Basic Actor-Critic
1. Sample $\{ \mathbf{s}_{i},\mathbf{a}_{i},\mathbf{s}_{i}' \}$ from $\pi_{\theta}(\mathbf{a}\mid \mathbf{s})$ (run policy).
2. Evaluate $y_{i}=r(\mathbf{s}_{i},\mathbf{a}_{i})+\gamma \hat{V}_{\phi}^{\pi}(\mathbf{s}_{i}')$.
3. Refit $\hat{V}$ to targets $\{ y_{i} \}$, minimizing $\mathcal{L}(\phi)$.
4. Evaluate $\hat{A}^{\pi}(\mathbf{s}_{i},\mathbf{a}_{i})=r(\mathbf{s}_{i},\mathbf{a}_{i})+\gamma \hat{V}_{\phi}^{\pi}(\mathbf{s}_{i}')-V_{\phi}^{\pi}(\mathbf{s}_{i})$.
5. Compute $\nabla_{\theta}J(\theta)\approx \sum_{i}\nabla_{\theta}\log \pi_{\theta}(\mathbf{a}_{i}\mid \mathbf{s}_{i})\hat{A}^{\pi}(\mathbf{s}_{i},\mathbf{a}_{i})$.
6. Update $\theta\leftarrow\theta+\alpha \nabla_{\theta}J(\theta)$.
7. Repeat!
### Online Actor-Critic
One may realize that, because we're just using transitions as training data, rather than whole trajectories, we don't need to use a whole trajectory or set of trajectories each iteration of model training.

The most extreme example of this is using *only one transition* to train the model each iteration. This actually allows use to have a fully online actor-critic algorithm that, during the simulation/in the environment, takes one transition, trains the model on that transition, and then uses the updated model to decide its next action.

This, however, has several issues.

- It is biased, as $\hat{V}$ is essentially "out of date" by one time step.
- The data is not i.i.d. anymore! The next time step's data is dependent on the current time step's data.
- Small batch size of $1$ is very volatile.

In practice, this can actually work, but only with multiple parallel workers and lots of hyperparameter tuning.
## On-Policy Actor-Critic
Can we improve the basic actor-critic algorithm? Yes!

- Better ways to estimate $\hat{A}^{\pi}$. (Improve the *critic*)
- Better ways to estimate $\nabla_{\theta}J(\theta)$. (Improve the *actor*)

So far, we've seen two methods of estimating $\nabla_{\theta}J(\theta)$. *Actor-critic* methods lower variance, but add some bias due to an *imperfect critic*, i.e. $\hat{V}_{\phi}^{\pi}$. *Policy-gradient* methods are entirely unbiased, but has higher variance due to it being a single sample estimate.

> [!question]- Which actor-critic RL objectives are biased, and which are unbiased?
> The actor-critic RL objectives here all (generally) take the form
> $$
> \nabla_{\theta}J(\theta)\approx \mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[ \sum_{t=1}^{H} \nabla_{\theta}\log \pi_{\theta}(\mathbf{a}_{t}\mid \mathbf{s}_{t})(\hat{Q}^{\pi}(\mathbf{s}_{t},\mathbf{a}_{t})-b) \right]
> $$
> for an estimator $\hat{Q}^{\pi}$ of the true $Q$-function (or the advantage function) and a baseline $b$. The key idea is that $\hat{Q}^{\pi}(\mathbf{s}_{t},\mathbf{a}_{t})$ must remain an unbiased estimator of the $Q$-function to keep the estimator unbiased. However, the baseline $b$ need only be independent of the current action $\mathbf{a}_{t}$ to keep the estimator unbiased.

> [!question]
> Is it possible to use $\hat{V}^{\pi}$ but still produce an unbiased estimate?

In fact, we can, with the following estimator. (Note that we are actually improving the *critic* here).
$$
\nabla_{\theta}J(\theta)\approx \frac{1}{N}\sum_{i=1}^{N} \sum_{t=1}^{H} \nabla_{\theta}\log \pi_{\theta}(\mathbf{a}_{t}^{(i)}\mid \mathbf{s}_{t}^{(i)})\left( \left( \sum_{t'=t}^{H} \gamma^{t'-t}r(\mathbf{s}_{t'}^{(i)},\mathbf{a}_{t'}^{(i)}) \right)-\hat{V}_{\phi}^{\pi}(\mathbf{s}_{t}^{(i)}) \right)
$$
which is essentially a compromise between the two methods, i.e. it's policy-gradient but the baseline is simply replaced by the value function. This produces a lower variance, unbiased estimator. (Sidenote: this is also a discounted, reward-to-go formulation).

> [!question] Why is this unbiased?
> As long as the baseline used doesn't depend on $\theta$, it remains unbiased. See [[Lecture 5]] for the proof of an unbiased baseline; it may be reused here.

However, it'd be nice if we could have a sort of "sliding scale" that determines the degree of mixing between the two methods, rather than only having this one combination option...
### Eligibility Traces and $n$-step Returns
Let $\hat{A}_{\text{C}}(\mathbf{s}_{t},\mathbf{a}_{t})$ denote the standard actor-critic estimator of advantage
$$
r(\mathbf{s}_{t},\mathbf{a}_{t})+\gamma \hat{V}_{\phi}^{\pi}(\mathbf{s}_{t+1})-\hat{V}_{\phi}^{\pi}(\mathbf{s}_{t})
$$
Let $\hat{A}_{\text{MC}}$ denote the standard Monte Carlo estimator of advantage
$$
\sum_{t=t'}^{\infty}\gamma^{t'-t}r(\mathbf{s}_{t'},\mathbf{a}_{t'})-\hat{V}_{\phi}^{\pi}(\mathbf{s}_{t})
$$
In RL problems, we'd naturally expect higher variance further in the future. Thus, ideally, we'd like to use a high variance, unbiased estimator for time steps closer to the present, and have variance decrease (while bias may increase) as time moves further into the future.

One way of achieving this is to use an **$n$-step return estimator**.
$$
\hat{A}_{n}^{\pi}(\mathbf{s}_{t},\mathbf{a}_{t})=\sum_{t'=t}^{t+n} \gamma^{t'-t}r(\mathbf{s}_{t'},\mathbf{a}_{t'})-\hat{V}_{\phi}^{\pi}(\mathbf{s}_{t})+\gamma^{n}\hat{V}_{\phi}^{\pi}(\mathbf{s}_{t+n})
$$
Essentially, the $n$-step estimator uses the Monte Carlo estimator for $n$ steps—unbiased, but high variance—and then *switches* to the actor-critic estimator for the remaining steps until the end of the horizon (possibly $\infty$)—high bias, low variance. This provides a *discrete* cutoff point at which we switch between the two estimators. $n>1$ commonly works better!

But, can we make a *continuous* averages of these two methods?
### Generalized Advantage Estimation (GAE)
The **generalized advantage estimator** $\hat{A}^{\pi}_{\text{GAE}}(\mathbf{s}_{t},\mathbf{a}_{t})$ is a *weighted average of $n$-step returns*.
$$
\hat{A}^{\pi}_{\text{GAE}}(\mathbf{s}_{t},\mathbf{a}_{t})=\sum_{n=1}^{\infty} w_{n}\hat{A}_{n}^{\pi}(\mathbf{s}_{t},\mathbf{a}_{t})
$$
where, typically, $w_{n}\propto\lambda^{n-1}$, so that $n$-step returns closer to the present are weighted more, therefore reducing variance. Moreover, it leads to an elegant simplification of the advantage estimator.
$$
\begin{align*}
\hat{A}^{\pi}_{\text{GAE}}&=r(\mathbf{s}_{t},\mathbf{a}_{t})+\gamma((1-\lambda)\hat{V}_{\phi}^{\pi}(\mathbf{s}_{t+1})+\lambda(r(\mathbf{s}_{t+1},\mathbf{a}_{t+1})+\dots )) \\
&= \sum_{t'=t}^{\infty} (\gamma\lambda)^{t'-t}\delta_{t'}
\end{align*}
$$
where $\delta_{t'}=r(\mathbf{s}_{t'},\mathbf{a}_{t'})+\gamma \hat{V}_{\phi}^{\pi}(\mathbf{s}_{t'+1})-\hat{V}_{\phi}^{\pi}(\mathbf{s}_{t'})$. In other words, the $\lambda$ behaves similarly to the discount factor $\gamma$. This expression may then be rewritten as a vastly *more efficient* ***recursive*** formula
$$
\begin{align*}
\hat{A}_{\text{GAE}}^{\pi}(\mathbf{s}_{t},\mathbf{a}_{t}) = \delta_{t'}+\gamma\lambda \hat{A}^{\pi}_{\text{GAE}}(\mathbf{s}_{t+1},\mathbf{a}_{t+1})
\end{align*}
$$

> [!info]
> This is the most popular advantage estimator used for policy gradient methods in modern reinforcement learning.

> [!question] is $\lambda$ the exact same as $\gamma$?
> No. $\gamma$ actually affects the RL objective; smaller $\gamma$ means more short-term prioritization. $\lambda$ only affects the accuracy (bias/variance) of your estimator, not your actual objective. Hence, they should be adjusted as two distinct hyperparameters.

> [!tip]
> In practice, the advantage is also centered/normalized.
## Off-Policy Actor-Critic
### On-Policy vs Off-Policy
On-policy actor-critic methods update the policy using data generated from the current policy, and are

- Computationally cheap
- Sample inefficient

while off-policy actor-critic methods update the policy using past data generated from old policies, and are

- Computationally expensive
- Sample efficient
### Replay Buffer
The key idea behind off-policy algorithms is the ***reuse*** *of data*. On-policy algorithms throw out all previous data with every iteration. Off-policy algorithms store previous data in a **replay buffer**.

Recall the online single-sample on-policy actor-critic method from [[#Online Actor-Critic|before]]. What if, instead of using the currently sampled transition to evaluate our policy, we *store* the current transition into our replay buffer for later use, and use a *minibatch* sampled from the replay buffer to evaluate our policy? Well, this would actually break the algorithm—the data in the replay buffer will no longer be *representative* of our current value estimator, and may instead be garbage from past, bad value estimators; thus, our policy evaluation may return garbage results.

Critically, though, <font color="#c0504d">if we estimate the *$Q$-function* instead of the value function</font>, this works! How?

First, consider why the value function method doesn't work. We have
$$
\begin{align*}
y_{i} &= r(\mathbf{s}_{i},\mathbf{a}_{i})+\gamma \hat{V}_{\phi}^{\pi}(\mathbf{s}_{i}'),\qquad(\mathbf{s}_{i},\mathbf{a}_{i},\mathbf{s}_{i}')\sim \bar{\pi} \\
\mathcal{L}(\phi) &= \frac{1}{2}\sum_{i=1}^{N} \lVert \hat{V}_{\phi}^{\pi}(\mathbf{s}_{i})-y_{i} \rVert ^{2} \\
V^{\pi}(\mathbf{s}_{i}) &\approx \mathbb{E}_{\mathbf{a}\sim \bar{\pi}(\mathbf{a}\mid \mathbf{s}_{i})}[r(\mathbf{s}_{i},\mathbf{a})+\gamma V^{\pi}(\mathbf{s}')]
\end{align*}
$$
However, we want $V^{\pi}(\mathbf{s}_{i})$ to be an expectation where $\mathbf{a}$ is sampled from $\pi_{\theta}(\mathbf{a}\mid \mathbf{s}_{i})$, not $\bar{\pi}$.

Now, let's look at using a $Q$-function instead. We have
$$
\begin{align*}
y_{i} &= r(\mathbf{s}_{i},\mathbf{a}_{i}) + \gamma \mathbb{E}_{\mathbf{a}'\sim \pi_{\theta}(\mathbf{a}'\mid \mathbf{s}_{i}')}[\hat{Q}_{\phi}^{\pi}(\mathbf{s}_{i}',\mathbf{a}')],\qquad(\mathbf{s}_{i},\mathbf{a}_{i},\mathbf{s}_{i}')\sim \bar{\pi} \\
\mathcal{L}(\phi) &= \frac{1}{2}\sum_{i=1}^{N} \lVert \hat{Q}_{\phi}^{\pi}(\mathbf{s}_{i})-y_{i} \rVert ^{2} \\
Q^{\pi}(\mathbf{s}_{i},\mathbf{a}_{i}) &\approx \mathbb{E}_{\mathbf{a}'\sim \pi_{\theta}(\mathbf{a}\mid \mathbf{s}_{i}')}[\hat{Q}^{\pi}_{\phi}(\mathbf{s}_{i}',\mathbf{a}')]
\end{align*}
$$
The key distinction is that, while the data is still sampled from an old policy, the expectation is computed over actions drawn from the current policy $\pi_{\theta}$. Why? Because we don't have to interact with the environment to sample from the latest policy and compute our $Q$-function, i.e. these are both represented by neural networks/models we are training. In contrast, the value function method must sample the actions from the old policy data; it cannot instead sample $\mathbf{a}'$ from the current policy because we're estimating $V^{\pi}(\mathbf{s}_{i})$, and estimating $V^{\pi}(\mathbf{s}_{i}')$ would require sampling a new transition/interacting with the environment, not reusing old policy data. (Because the estimate would be based on the next state, so, without state transition probabilities, one would need to simulate taking action $\mathbf{a}'$ from state $\mathbf{s}'$ in the environment to produce the next state, and then $V^{\pi}$ of this state may be computed).

This new algorithm needs a few more tweaks to finish it off.

1. Sample $(\mathbf{s}_{i},\mathbf{a}_{i},\mathbf{s}_{i}')$ using $\pi_{\theta}$ with one step and store in the replay buffer $\mathcal{R}$.
2. Load a minibatch and evaluate $y_{i}=r(\mathbf{s}_{i},\mathbf{a}_{i})+\mathbb{E}_{\mathbf{a}'\sim \pi_{\theta}(\mathbf{a}'\mid \mathbf{s}_{i}')}[\hat{Q}_{\phi}^{\pi}(\mathbf{s}_{i}',\mathbf{a}_{i}')]$ for each transition.
3. Update $\phi$ using $\nabla_{\phi}\sum_{i=1}^{B}\lVert \hat{Q}_{\phi}^{\pi}(\mathbf{s}_{i},\mathbf{a}_{i})-y_{i} \rVert^{2}$, where $B$ is the minibatch size.
4. Reuse the above trick to now *sample $\mathbf{a}_{i}$* (rather than $\mathbf{a}_{i}'$) from the latest policy, i.e. $\mathbf{a}_{i}^{\pi}\sim \pi_{\theta}(\mathbf{a}\mid \mathbf{s}_{i})$, and compute $\nabla_{\theta}J(\theta)=\frac{1}{N}\sum_{i}\nabla_{\theta}\log \pi_{\theta}(\mathbf{a}_{i}^{\pi}\mid \mathbf{s}_{i})\hat{A}^{\pi}(\mathbf{s}_{i},\mathbf{a}_{i}^{\pi})$.
5. $\theta\leftarrow\theta+\alpha \nabla_{\theta}J(\theta)$.
6. Repeat

> [!tip]
> In practice, $\hat{Q}^{\pi}_{\phi}$ is used in place of $\hat{A}^{\pi}$ because variance only really matters when sampling is expensive, as using more data/samples naturally lowers variance, and therefore lowering variance through techniques like baselines is not as necessary for this off-policy actor-critic $Q$ method. (And excluding variance-lowering methods results in lower implementation complexity).

> [!question] Is it an issue that the state distribution doesn't match, i.e. the state distribution is sampled from old policies?
> Theoretically, yes, but in practice, with a sufficiently high capacity model, this poses no issues, as it doesn't really hurt the model.
### Reparametrization Trick
Previously, with [[Lecture 5#Direct Policy Differentiation|direct policy differentiation]], we had to use samples from our policy used with the environment to compute the gradient $\nabla_{\theta}J(\theta)$. However, with this off-policy method, to estimate $\nabla_{\theta}J(\theta)$, we are estimating, essentially,
$$
\nabla_{\theta}\mathbb{E}_{\mathbf{a}\sim \pi_{\theta}(\mathbf{a}\mid \mathbf{s}_{i})}[\hat{Q}_{\phi}^{\pi}(\mathbf{s}_{i},\mathbf{a})]
$$
Crucially, $\hat{Q}$ and $\pi_{\theta}$ are both neural networks/models that we are training, and thus can evaluate <font color="#c0504d">without involving system dynamics/the environment at all</font>! Therefore, we can actually compute $\frac{\textrm{d}}{\textrm{d} \mathbf{a} }\hat{Q}_{\phi}^{\pi}(\mathbf{s}_{i},\mathbf{a})$ with backprop, which produces a better gradient estimator.

> [!warning]
> This only works for continuous, parametric action distributions and when $\hat{Q}$ is differentiable.

How can we utilize this? WLOG, assume $\pi_{\theta}(\mathbf{a}\mid \mathbf{s})=\mathcal{N}(\mu_{\theta}(\mathbf{s}),\sigma_{\theta}(\mathbf{s}))$. Let $\epsilon \sim \mathcal{N}(0,1)$. Then, one may note that $\mathbf{a}=\mu_{\theta}(\mathbf{s})+\sigma_{\theta}(\mathbf{s})\epsilon$. This allows us to approximate
$$
\begin{align*}
\mathbb{E}_{\mathbf{a}\sim \pi_{\theta}(\mathbf{a}\mid \mathbf{s})}[Q(\mathbf{s},\mathbf{a})] &= \mathbb{E}_{\epsilon\sim \mathcal{N}(0,1)}[Q(\mathbf{s},\mu_{\theta}(\mathbf{s})+\sigma_{\theta}(\mathbf{s})\epsilon)] \\
&\approx Q(\mathbf{s},\mu_{\theta}(\mathbf{s})+\sigma_{\theta}(\mathbf{s})\epsilon) \\
\nabla_{\theta}\mathbb{E}_{\mathbf{a}\sim \pi_{\theta}(\mathbf{a}\mid \mathbf{s})}[Q(\mathbf{s},\mathbf{a})] &\approx \nabla_{\theta}Q(\mathbf{s},\mu_{\theta}(\mathbf{s})+\sigma_{\theta}(\mathbf{s})\epsilon)
\end{align*}
$$
and the last expression is easy to differentiate with an autograd library. This is known as the **reparametrized gradient estimator**.

> [!tip] Finally, we have an actual, practical algorithm for RL :)

> [!question] What??
> See the last part of [[Lecture 12#Reparametrization Trick|this section]] of Lecture 12 that I wrote after I did some more research into the reparametrization trick. Note that it may help to know what variational inference is, i.e. [[Lecture 11]].
## Summary
A collection of key formulas.
### On-policy Actor/Advantage Estimation
$$
\begin{align*}
\hat{A}_{\text{MC}}^{\pi}(\mathbf{s}_{t},\mathbf{a}_{t})&= \sum_{t'=t}^{H} \gamma^{t'-t}r(\mathbf{s}_{t'},\mathbf{a}_{t'})-\hat{V}_{\phi}^{\pi}(\mathbf{s}_{t}) && \text{Monte Carlo} \\
\hat{A}_{\text{C}}^{\pi}(\mathbf{s}_{t},\mathbf{a}_{t}) &= r(\mathbf{s}_{t},\mathbf{a}_{t})+\gamma \hat{V}_{\phi}^{\pi}(\mathbf{s}_{t+1})-\hat{V}_{\phi}^{\pi}(\mathbf{s}_{t}) && \text{Bootstrapped} \\
\hat{A}_{\text{n}}^{\pi}(\mathbf{s}_{t},\mathbf{a}_{t}) &= \sum_{t'=t}^{t+n-1} \gamma^{t'-t}r(\mathbf{s}_{t'},\mathbf{a}_{t'})+\gamma^{n}\hat{V}_{\phi}^{\pi}(\mathbf{s}_{t+1})-\hat{V}_{\phi}^{\pi}(\mathbf{s}_{t}) && n\text{-step} \\
\hat{A}_{\text{GAE}}^{\pi}(\mathbf{s}_{t},\mathbf{a}_{t}) &= \frac{1-\lambda}{1-\lambda^{H-t}} \sum_{n=1}^{H-t}\lambda^{n-1}\hat{A}_{n}^{\pi}(\mathbf{s}_{t},\mathbf{a}_{t}) && \text{Generalized Advantage} \\ 
&= \sum_{t'=t}^{\infty} (\gamma\lambda)^{t'-t}\hat{A}_{C}^{\pi}(\mathbf{s}_{t'},\mathbf{a}_{t'}) \\
&= \hat{A}_{\text{C}}^{\pi}(\mathbf{s}_{t},\mathbf{a}_{t}) + \gamma\lambda \hat{A}_{\text{GAE}}^{\pi}(\mathbf{s}_{t+1},\mathbf{a}_{t+1})
\end{align*}
$$
Note that $\frac{1-\lambda}{1-\lambda^{H-t}}=\frac{1}{1+\lambda+\dots+\lambda^{H-t-1}}$ in the GAE expression is the aforementioned *normalizing factor*.
<div style="page-break-after: always;"></div>

