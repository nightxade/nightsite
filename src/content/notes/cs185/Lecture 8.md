# Lecture 8: $Q$-Learning in Practice
## Target Networks
So, why doesn't $Q$-learning work if we just implement what we discussed in [[Lecture 7]]? Well, it's not really gradient descent. Recall that the target values are associated with a *stop-gradient* operator, i.e.
$$
\theta\leftarrow \phi-\alpha \frac{\mathrm{d} Q_{\theta} }{\mathrm{d} \theta } (\mathbf{s}_{i},\mathbf{a}_{i})(Q_{\theta}(\mathbf{s}_{i},\mathbf{a}_{i})-[r(\mathbf{s}_{i},\mathbf{a}_{i})+\gamma \max _{\mathbf{a}'}Q_{\theta}(\mathbf{s}_{i}',\mathbf{a}_{i}')]_{\times })
$$
therefore, $Q$-learning is not gradient descent because it doesn't have a well-defined objective, and thus it does not necessarily converge!

Recall that $Q$-learning is a type of fitted $Q$-iteration algorithm. The primary difference, however, is that $Q$-learning takes only *one step* between each gradient update. In contrast, $Q$-iteration collects a large, static dataset of transitions, and then performs a *full regression*, not just a single gradient step, on the entire dataset. In essence, $Q$-learning chases a moving target during optimization, while $Q$-iteration "freezes" the target $y_{i}$ values and performs essentially standard supervised learning each step to find the new $\theta$.

In some sense, one can think of this aspect of $Q$-iteration as storing a "frozen copy" of the model for $Q_{\theta}$ when learning the next value of $\theta$. We can attempt to achieve this same "static" property of $Q$-iteration in $Q$-learning with this frozen copy idea—we separately store a model $Q_{\bar{\theta}}$ where $\bar{\theta}\leftarrow\theta$ every $X$ iterations, e.g. $X=10^{4}$. Then, we use this frozen $Q$-function to calculate the targets $y_{i}$, i.e.
$$
y_{i}=r(\mathbf{s}_{i},\mathbf{a}_{i})+\gamma \max _{\mathbf{a}'}Q_{\bar{\theta}}(\mathbf{s}_{i}',\mathbf{a}')
$$
this is known as **$Q$-learning with target networks**, and allows us to stabilize learning a bit and "slow the moving target."

This has some drawbacks, though. In particular, $Q_{\bar{\theta}}$ lags behind $Q_{\theta}$, and may produce a fairly inaccurate representation of the true $Q$-function.
### Deep $Q$-Learning Network (DQN)
Famous foundational reinforcement learning algorithm—it's just generalized $Q$-learning with a replay buffer and target network.

1. Save target network parameters: $\bar{\theta}\leftarrow\theta$.
	2. Collect dataset $\{ (\mathbf{s}_{i},\mathbf{a}_{i},\mathbf{s}_{i}') \}$ with some policy, add to buffer $\mathcal{R}$.
		3. Sample batch $(\mathbf{s}_{i},\mathbf{a}_{i},\mathbf{s}_{i}')\sim \mathcal{R}$.
		4. Update $\theta\leftarrow\theta-\alpha \sum_{i}\frac{\mathrm{d} Q_{\theta} }{\mathrm{d} \theta }(\mathbf{s}_{i},\mathbf{a}_{i})(Q_{\theta}(\mathbf{s}_{i},\mathbf{a}_{i})-[r(\mathbf{s}_{i},\mathbf{a}_{i})+\gamma \max_{\mathbf{a}'}Q_{\bar{\theta}}(\mathbf{s}_{i}',\mathbf{a}_{i}')])$.

where the indentation denotes nested loops. For DQN, the innermost loop iterates $1$ time and the loop starting on step 2 iterates $1$ times.
### Alternatives to Target Networks
Target networks, notably are **discrete** updates, in that they update only every $X$ steps. This produces bursty changes in the learning algorithm. In practice, this isn't usually an issue; but, if necessary, one solution is to use **[[8. Optimization for Training Deep Models#Polyak Averaging|Polyak averaging]]**, in which we update $\bar{\theta}\leftarrow \tau\bar{\theta}+(1-\tau)\theta$ (e.g. $\tau=0.999$).
## More Efficient $Q$-Learning
Generalized $Q$-learning with a replay buffer and target network can be represented as three different processes that essentially run at their own rates.

1. Data collection (also evicting old data).
2. Update target $\bar{\theta}$.
3. $Q$-function regression.

Here's the speeds for some different types of $Q$-learning.
- In online $Q$-learning, data is evicted immediately, and all three processes run at the same speed.
- In DQN, process 1 and process 3 run at the same speed, while process 2 is slow.
- In fitted $Q$-iteration, process 3 runs faster than process 2, which runs faster than process 1

Also, here's how we'll describe the generalized $Q$-learning algorithm with a replay buffer and target network.
1. Save target network parameters: $\bar{\theta}\leftarrow\theta$.
	
	2. Collect dataset $\{ (\mathbf{s}_{i},\mathbf{a}_{i},\mathbf{s}_{i}') \}$ with some policy, add to buffer $\mathcal{R}$. (Loop $N$ times). 
		
		3. Sample batch $(\mathbf{s}_{i},\mathbf{a}_{i},\mathbf{s}_{i}')\sim \mathcal{R}$. (Loop $K$ times).
		4. Update $\theta\leftarrow\theta-\alpha \sum_{i}\frac{\mathrm{d} Q_{\theta} }{\mathrm{d} \theta }(\mathbf{s}_{i},\mathbf{a}_{i})(Q_{\theta}(\mathbf{s}_{i},\mathbf{a}_{i})-[r(\mathbf{s}_{i},\mathbf{a}_{i})+\gamma \max_{\mathbf{a}'}Q_{\bar{\theta}}(\mathbf{s}_{i}',\mathbf{a}_{i}')])$.

So, which choices of $N,K$ are most efficient?
### $K$
Let's first focus on $K$. This is also known as the **update-to-data (UTD)** ratio. A higher $K$ increases learning speed, as more updates are made for every data collection step, but this is more computationally expensive. Moreover, an excessively high $K$ will cause overfitting to the current set of data. In fact, $K\geq10$ is generally already a risky choice!
### $N$
Now, consider $N$, the ratio of dataset collection steps to target network update steps. A lower $N$ increases learning speed, as fewer steps between target network updates keeps the target network more up-to-date. Generally speaking, though, $N\sim[10^{3},10^{4}]$ as lower values creates a lot of instability.
### $n$-step Returns?
In essence, we can apply $n$-step returns to the target values themselves!
$$
y_{t}^{(i)}=\sum_{t'=t}^{t+n} \gamma^{t'-t}r(\mathbf{s}_{t'}^{(i)},\mathbf{a}_{t'}^{(i)}) + \gamma^{n}\max _{\mathbf{a}_{t+n}^{(i)}} Q_{\bar{\theta}}(\mathbf{s}_{t+n}^{(i)},\mathbf{a}_{t+n}^{(i)})
$$
This *increases* learning speed because the target values are now calculated from a mix of the up-to-date reward values from a trajectory and the out-of-date target network. However, this has a major limitation—because we're learning with a *trajectory* that originates from older data from a worse policy, the target values are lower than they should be! Thus, the off-policy nature of $Q$-learning makes this $n$-step returns modification *biased*. Hence, people will typically use small $n$ values if applying this technique.
## Overestimation in $Q$-Learning
For naive DQN, the $Q$-functions tend to **overestimate** the actual returns. The problem is the $\max$ function in calculating the target value for target networks, i.e.
$$
y_{i}=r(\mathbf{s}_{i},\mathbf{a}_{i})+\gamma\, \boxed{ \max }_{\mathbf{a}'_{i}}Q_{\bar{\theta}}(\mathbf{s}_{i},\mathbf{a}_{i})
$$
Why? Because it essentially selects the $Q$-functions with the *largest positive error*.

To form some intuition, consider two two random variables $X_{1},X_{2}$. Then, $\mathbb{E}[\max(X_{1},X_{2})]\geq \max(\mathbb{E}[X_{1}],\mathbb{E}[X_{2}])$. In other words, taking the $\max$ of the $Q_{\hat{\theta}}$ function **systematically overestimates** the maximum of the expected values of the actions.
### Double $Q$-Learning
We note that
$$
\max _{\mathbf{a}'}Q_{\bar{\theta}}(\mathbf{s}',\mathbf{a}') = Q_{\bar{\theta}}(\mathbf{s}',\underset{\mathbf{a}'}{\arg\max}\ Q_{\bar{\theta}}(\mathbf{s}',\mathbf{a}') )
$$
the key idea of double $Q$-learning is to use **two distinct models** in the RHS expression—one to choose the action (inner $Q$) and one to evaluate the value (outer $Q$). This helps eliminate the overestimation error because, if both models have *different noise patterns*, then the action chosen to maximize the inner $Q$ (and thus possesses the largest positive error) does not necessarily correspond to the value with the largest positive error in the outer $Q$.

*Classic* double $Q$-learning was thus modeled as
$$
\begin{align*}
Q_{\theta_{A}}\leftarrow r+\gamma Q_{\theta_{B}}(\mathbf{s}',\underset{\mathbf{a}'}{\arg\max}\ Q_{\theta_{A}}(\mathbf{s}',\mathbf{a}')) \\
Q_{\theta_{B}}\leftarrow r+\gamma Q_{\theta_{A}}(\mathbf{s}',\underset{\mathbf{a}'}{\arg\max}\ Q_{\theta_{B}}(\mathbf{s}',\mathbf{a}'))
\end{align*}
$$
*In practice*, though, there is an easier implementation for double $Q$-learning. One may note that DQN itself already has two $Q$-functions: the target network and the current $Q$-function. Thus, double $Q$-learning is modeled as
$$
y=r+\gamma Q_{\bar{\theta}}(\mathbf{s}',\underset{\mathbf{a}'}{\arg\max}\ Q_{\theta}(\mathbf{s}',\mathbf{a}'))
$$
> [!question] $\theta \sim\bar{\theta}$??
> Aren't $\theta$ and $\bar{\theta}$ correlated? Yes, but they are distinct enough that this proves very effective in practice. And this method is just much easier to implement so :)
### Generalized Double $Q$-Learning
Note that $\arg\max_{\mathbf{a}'}Q_{\theta}(\mathbf{s}',\mathbf{a}')$ is essentially equivalent to taking the action according to a greedy policy, i.e. $Q_{\bar{\theta}}(\mathbf{s}',\arg\max_{\mathbf{a}'}Q_{\theta}(\mathbf{s}',\mathbf{a}'))=\mathbb{E}_{\mathbf{a}'\sim \pi_{\theta}(\mathbf{a}'\mid \mathbf{s}')}[Q_{\bar{\theta}}(\mathbf{s}',\mathbf{a}')]$. In other words,
$$
y=r+\gamma \mathbb{E}_{\mathbf{a}'\sim \pi_{\theta}(\mathbf{a}'\mid \mathbf{s}')}[Q_{\bar{\theta}}(\mathbf{s}',\mathbf{a}')]
$$
Occasionally, if overestimation is a problem even after double $Q$-learning, some practitioners implement **clipped** double $Q$-learning.
$$
y=r+\gamma \mathbb{E}_{\mathbf{a}'\sim \pi_{\theta}(\mathbf{a}',\mathbf{s}')}[\min _{i\in \{ 1,2 \}}Q_{\bar{\theta}_{j}}(\mathbf{s}',\mathbf{a}')]
$$
where we now train an **ensemble** of two different target networks $Q_{\bar{\theta}_{1}}$ and $Q_{\bar{\theta}_{2}}$. (You may have any number of target networks, but it gets increasingly expensive). This can, however, cause *underestimates* for $Q$-learning, though it's often still practical for actor-critic!
## Tips for $Q$-Learning
### Practical Tips
- $Q$-learning takes time to stabilize
	- Test on easier tasks first to validate implementation
	- Be patient when training
- Large replay buffers improve stability
- Start with high exploration (epsilon) and gradually reduce
### Advanced Tips
- Bellman error gradients can be big; clip gradients or use Huber loss
- Double $Q$-learning is very effective in practice
- $n$-step returns can help a lot, but be careful
- Schedule exploration/learning rate or just use Adam
- Run multiple times with different seeds
## Back to Continuous Actions
### $Q$-learning with continuous actions
With continuous actions, choosing the value-maximizing action value is hard and inefficient. This is particularly problematic for when we're evaluating the target values, as this occurs in the innermost training loop. There are a couple of ways to solve this.
#### Solution 1: Stochastic Optimization
The simplest solution is to approximate $\max_{\mathbf{a}}Q(\mathbf{s},\mathbf{a})$ with some samples, i.e.
$$
\max _{\mathbf{a}}Q(\mathbf{s},\mathbf{a})\approx \max \{ Q(\mathbf{s},\mathbf{a}_{1}),\dots ,Q(\mathbf{s},\mathbf{a}_{N}) \}
$$
this is efficiently *parallelizable* and easy to implement, but unfortunately lacking in accuracy. We can do better.
- Cross-Entropy Method (CEM)
- CMA-ES

these stochastic optimizers work decently up to about 40 dimensions.

> [!info]
> Very popular improvement to offline actor-critic methods.

Such methods are also known as **sample and rank** or **rejection sampling**.
#### Solution 2: Learn Approximate Maximizer
The classic example is **deep deterministic policy gradient (DDPG)**, and is essentially *deterministic actor critic*. The idea is to train another model $\mu_{\theta}(\mathbf{s})$ to approximate $\arg\max_{\mathbf{a}}Q_{\phi}(\mathbf{s},\mathbf{a})$, where we find $\theta\leftarrow\arg\max_{\theta}Q_{\phi}(\mathbf{s},\mu_{\theta}(\mathbf{s}))$. This is found by using the following
$$
\frac{\textrm{d} Q_{\phi} }{\textrm{d} \theta } = \frac{\textrm{d} \mathbf{a} }{\textrm{d} \theta } \frac{\textrm{d} Q_{\phi} }{\textrm{d} \mathbf{a} } 
$$
which is essentially the same as the [[Lecture 6#Reparametrization Trick|reparametrization trick]] from Lecture 6.

The new target value is thus
$$
y_{j}= r_{j}+\gamma Q_{\bar{\phi}}(\mathbf{s}_{j}',\mu_{\bar{\theta}}(\mathbf{s}_{j}'))
$$
> [!question] Deterministic Actor-Critic?
> $\mu_{\theta}$ is the actor, $Q_{\phi}$ is the critic. Deterministic because there is no expected value on $Q_{\phi}$ over sampled actions from the policy.
## Some $Q$-Learning Theory
### Value Iteration
Let's return to value iteration.

1. $Q(\mathbf{s},\mathbf{a})\leftarrow r(\mathbf{s},\mathbf{a})+\mathbb{E}[V(\mathbf{s}')]$.
2. $V(\mathbf{s})\leftarrow \max_{\mathbf{a}}Q(\mathbf{s},\mathbf{a})$.
3. Repeat.

Does value iteration converge?

Let $\mathcal{B}$ be an operator such that, for a vector $V$ composed of the values of the different states,
$$
\mathcal{B}V=\max _{\mathbf{a}}r_{\mathbf{a}} + \gamma \mathcal{T}_{\mathbf{a}}V
$$
where $r$ is the stacked vector of rewards at all states for action $\mathbf{a}$ and $\mathcal{T}_{\mathbf{a}}$ is the matrix of transitions for action $\mathbf{a}$, i.e. $\mathcal{T}_{\mathbf{a},i,j}=p(\mathbf{s}'=i\mid \mathbf{s}=j,\mathbf{a})$. This is essentially value iteration written out with just an operator $\mathcal{B}$ and a value vector $V$, i.e.

1. $V\leftarrow \mathcal{B}V$
2. Repeat

We note that $V^{*}$, the vector representing the *optimal* value function, is a **fixed point** of $\mathcal{B}$, i.e. $V^{*}=\mathcal{B}V^{*}$. We can prove that $V^{*}$ always exists, is always unique, and is always optimal. The way we prove that $\mathcal{B}$ converges to $V^{*}$ is by showing that $\mathcal{B}$ is a **contraction**. That is, for any vectors $V,\bar{V}$, $\lVert \mathcal{B}V-\mathcal{B}\bar{V} \rVert_{\infty}\leq\gamma \lVert V-\bar{V} \rVert_{\infty}$. In other words, the gap always gets smaller by at least a factor of $\gamma$. We will not prove this now, just take my word :)

If we choose $\bar{V}=V^{*}$, we have that $\lVert \mathcal{B}V-\mathcal{B}V^{*} \rVert_{\infty}=\lVert \mathcal{B}V-V^{*} \rVert_{\infty}\leq\gamma \lVert V-V^{*} \rVert_{\infty}$, or that the operator always contracts the gap. Thus, value iteration converges. <p style="margin: 0px 0px 0px 0px;" align="right"> $\square$ </p>
### Non-Tabular Value Function Learning
Non-tabular value function learning, e.g. fitted value iteration, is a bit trickier, because we train a model that has at least *some error*. However, we can interpret the regression performed by fitted value iteration as a *projection* (in the L2 norm, due to MSE) onto the space of functions representable by the model.
$$
V'\leftarrow \underset{V'\in\Omega}{\arg\min}\ \frac{1}{2}\sum \lVert V'(\mathbf{s})-(\mathcal{B}V)(\mathbf{s}) \rVert^{2}
$$
where $\Omega$ represents the space of representable value functions. In essence, each time we update $V\leftarrow \mathcal{B}V$, we first project $\mathcal{B}V$ back into $\Omega$. To simplify this, we define a new operator
$$
\Pi:\Pi V=\underset{V'\in\Omega}{\arg\min}\ \frac{1}{2}\sum \lVert V'(\mathbf{s})-V(\mathbf{s}) \rVert ^{2}
$$
so we can write fitted value iteration as

1. $V\leftarrow \Pi \mathcal{B}V$
2. Repeat

Now, $\mathcal{B}$ is a contraction w.r.t $\infty$ norm, and $\Pi$ is a contraction w.r.t $\ell_{2}$ norm. Yet, because they are contractions only w.r.t different norms, $\Pi \mathcal{B}$ is *not* a contraction! Thus, while value iteration does converge, fitted value iteration **does not converge**. (Though, with deep RL, $\Omega$ is much larger in practice, and therefore works well). <p style="margin: 0px 0px 0px 0px;" align="right"> $\square$ </p>
### Fitted $Q$-Iteration, Actor-Critic
We can do the same for $Q$-iteration (and online $Q$-learning) and actor-critic, and find that they too are not guaranteed convergence for essentially the same reason.
### Conclusions
- Fitted value-based methods ($Q$-learning, $Q$-function actor-critic) are frequently *unstable*!
	- In practice, not guaranteed convergence
	- Requires more hyperparameter tuning
- Some tricks are very helpful though...
	- Replay buffers
	- Target networks
<div style="page-break-after: always;"></div>

