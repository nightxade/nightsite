# Lecture 18: Offline RL Algorithms
## Policy Constraint Methods
### Explicit Policy Constraint Methods
To add policy constraints to, e.g., $Q$-function actor-critic, we can modify the <font color="#c0504d">actor objective</font>
$$
\begin{align*}
\theta &\leftarrow \underset{\theta}{\arg\max}\ \sum_{\mathbf{s}_{i}\in \mathcal{D}}\mathbb{E}_{\mathbf{a}\sim \pi_{\theta}(\mathbf{a}\mid \mathbf{s}_{i})}[Q(\mathbf{s}_{i},\mathbf{a}_{i})]+\lambda \log \pi_{\theta}(\mathbf{a}_{i}\mid \mathbf{s}_{i}) \\
\theta &\leftarrow \underset{\theta}{\arg\max}\ \sum_{\mathbf{s}_{i}\in \mathcal{D}}\mathbb{E}_{\mathbf{a}\sim \pi_{\theta}(\mathbf{a}\mid \mathbf{s}_{i})}[Q(\mathbf{s}_{i},\mathbf{a}_{i})+\lambda \log \pi_{\beta}(\mathbf{a}\mid \mathbf{s}_{i})]+\lambda \mathcal{H}(\pi_{\theta}(\mathbf{a}\mid \mathbf{s}_{i}))
\end{align*}
$$
for the forward and reverse KL divergence, respectively. Generally, this is not the best way to add policy constraints, but it can work well if done right.

We can also instead modify the <font color="#c0504d">reward function</font>. Typically, this is done with reverse KL + MaxEnt RL.
$$
\begin{align*}
\hat{r}(\mathbf{s},\mathbf{a})&=r(\mathbf{s},\mathbf{a})-\lambda D_{\text{KL}} \\
&= r(\mathbf{s},\mathbf{a})+\lambda \log \pi_{\beta}(\mathbf{a}\mid \mathbf{s})
\end{align*}
$$
This is particularly interesting because it also accounts for *future divergence*.

Now, we can construct some real offline actor-critic algorithms.

Here's a **BRAC-like**, behavior regularized actor critic, algorithm. (Behavior Regularized Offline Reinforcement Learning, Wu et al.)

1. Evaluate $y_{i}=r(\mathbf{s}_{i},\mathbf{a}_{i})+\gamma \mathbb{E}_{\mathbf{a}'\sim \pi_{\theta}(\mathbf{a}'\mid \mathbf{s}_{i}')}[\hat{Q}_{\phi}^{\pi}(\mathbf{s}_{i}',\mathbf{a}_{i})-\lambda D(\pi_{\theta}(\cdot \mid \mathbf{s}_{i}'),\pi_{\beta}(\cdot \mid\mathbf{s}_{i}'))]$. (modifying reward)
2. Update $\phi$ using $\nabla_{\phi}\sum_{i=1}^{B}\lVert \hat{Q}_{\phi}^{\pi}(\mathbf{s}_{i},\mathbf{a}_{i})-y_{i} \rVert^{2}$.
3. $J(\theta)=\sum_{i}\mathbb{E}_{\mathbf{a}\sim \pi_{\theta}(\mathbf{a}\mid \mathbfit{s},_{i})}[\hat{Q}_{\phi}^{\pi}(\mathbf{s}_{i},\mathbf{a})]-\lambda \mathcal{D}(\pi_{\theta}(\cdot \mid \mathbf{s}_{i}),\pi_{\beta}(\cdot \mid \mathbf{s}_{i}))$. (modifying actor objective)
4. $\theta\leftarrow\theta+\alpha \nabla_{\theta}J(\theta)$.

And here's an **AC+BC-like**, actor critic + behavioral cloning?, algorithm.

1. Evaluate $y_{i}=r(\mathbf{s}_{i},\mathbf{a}_{i})+\gamma \mathbb{E}_{\mathbf{a}'\sim \pi_{\theta}(\mathbf{a}'\mid \mathbf{s}_{i}')}[\hat{Q}_{\phi}^{\pi}(\mathbf{s}_{i}',\mathbf{a}_{i})]$.
2. Update $\phi$ using $\nabla_{\phi}\sum_{i=1}^{B}\lVert \hat{Q}_{\phi}^{\pi}(\mathbf{s}_{i},\mathbf{a}_{i})-y_{i} \rVert^{2}$.
3. $J(\theta)=\sum_{i}\mathbb{E}_{\mathbf{a}\sim \pi_{\theta}(\mathbf{a}\mid \mathbfit{s},_{i})}[\hat{Q}_{\phi}^{\pi}(\mathbf{s}_{i},\mathbf{a})]+\lambda \log \pi_{\theta}(\mathbf{a}_{i}\mid \mathbf{s}_{i})$. (modifying actor objective with forward KL)
4. $\theta\leftarrow\theta+\alpha \nabla_{\theta}J(\theta)$.
### Implicit Policy Constraint Methods
> [!info]
> From papers like Advantage-Weighted Regression (Peng et al.) and Accelerating Online Reinforcement Learning with Offline Datasets (Nair et al.). (Both advised by Levine orz).

Implicit policy constraint methods do not add an explicit regularizer term.

Consider, for instance, finding the optimal policy $\pi_{\text{new}}$ over expectation of a $Q$-function, subject to a reverse KL divergence constraint.
$$
\begin{align*}
\pi^{*} &= \underset{\pi}{\arg\max}\ \mathbb{E}_{\mathbf{a}\sim \pi(\mathbf{a}\mid \mathbf{s})}[Q(\mathbf{s},\mathbf{a})],\ D_{\text{KL}}(\pi \parallel \pi_{\beta})\leq \epsilon
\end{align*}
$$
Via classical convex optimization techniques, we can derive an exact, closed-form answer. (Note: this is usually not practical due to the size of the action space).
$$
\begin{align*}
\pi^{*}(\mathbf{a}\mid \mathbf{s}) &= \frac{1}{Z(\mathbf{s})}\pi_{\beta}(\mathbf{a}\mid \mathbf{s})\exp\left( \frac{1}{\lambda}A^{\pi}(\mathbf{s},\mathbf{a}) \right)
\end{align*}
$$
Now, the idea is that we can use **importance sampling** to effectively approximate samples from $\pi^{*}$ using samples from $\pi_{\beta}$, and then use **behavioral cloning** (supervised learning) to learn from the samples of $\pi^{*}$. In particular, we approximate via *importance-weighted maximum likelihood*, i.e. we produce an update rule
$$
\pi_{\text{new}}(\mathbf{a}\mid \mathbf{s}) = \underset{\pi}{\arg\max}\ \mathbb{E}_{(\mathbf{s},\mathbf{a})\sim \pi_{\beta}}\left[ \log \pi(\mathbf{a}\mid \mathbf{s}) \underbrace{ \frac{1}{Z(\mathbf{s})}\exp\left( \frac{1}{\lambda}A^{\pi_{\text{old}}}(\mathbf{s},\mathbf{a}) \right) }_{ w(\mathbf{s},\mathbf{a}) } \right]
$$
In other words, maximizing the objective
$$
J(\theta) = \mathbb{E}_{(\mathbf{s},\mathbf{a})\sim \pi_{\beta}} \log \pi_{\theta}(\mathbf{a}\mid \mathbf{s})\exp(A^{\pi_{\theta}}(\mathbf{s},\mathbf{a}))
$$
where all terms not dependent on $\theta$ were removed, is equivalent to learning a policy $\pi_{\theta}$ that behaviorally clones the optimal policy $\pi^{*}$ under the reverse KL divergence constraint.

This may translate to an **AWAC-like**, advantage-weighted actor-critic, algorithm. (Nair et al.)

1. Evaluate $y_{i}=r(\mathbf{s}_{i},\mathbf{a}_{i})+\gamma \mathbb{E}_{\mathbf{a}'\sim \pi_{\theta}(\mathbf{a}'\mid \mathbf{s}_{i}')}[\hat{Q}_{\phi}^{\pi}(\mathbf{s}_{i}',\mathbf{a}_{i}')]$.
2. Update $\phi$ using $\nabla_{\phi}\sum_{i=1}^{B}\lVert \hat{Q}_{\phi}^{\pi}(\mathbf{s}_{i},\mathbf{a}_{i})-y_{i} \rVert^{2}$.
3. Update $\psi$ using $\nabla_{\psi}\sum_{i=1}^{B}\lVert \hat{V}_{\psi}^{\pi}(\mathbf{s}_{i})-\mathbb{E}_{\mathbf{a}\sim \pi_{\theta}(\mathbf{a}\mid \mathbf{s}_{i})}[\hat{Q}_{\phi}^{\pi}(\mathbf{s}_{i},\mathbf{a})] \rVert^{2}$. (train $\hat{V}_{\psi}^{\pi}$)
4. $J(\theta)=\sum_{i}\log \pi_{\theta}(\mathbf{a}_{i}\mid \mathbf{s}_{i})\exp(\hat{Q}_{\phi}^{\pi}(\mathbf{s}_{i},\mathbf{a}_{i})-\hat{V}_{\psi}^{\pi}(\mathbf{s}_{i}))$.
5. $\theta\leftarrow\theta+\alpha \nabla_{\theta}J(\theta)$.

Note that you can also approximate $\hat{V}^{\pi}(\mathbf{s}_{i})\approx Q_{\phi}^{\pi}(\mathbf{s}_{i},\mathbf{a})$ where $\mathbf{a}\sim \pi_{\theta}(\mathbf{a}\mid \mathbf{s}_{i})$ if you don't want to train a separate neural network to learn $\hat{V}^{\pi}$.

This approach is nice for its simplicity; however, it is limited primarily by the fact that, in copying the samples generated by the behavioral policy, it is slow to learn to avoid bad actions.
## Implicit $Q$-Learning (IQL)
> [!info]
> Paper: Offline Reinforcement Learning with Implicit Q-Learning (Kostrikov et al.). (Also Levine).

Here's a thought: what if we simply <font color="#c0504d">avoided all out-of-distribution actions</font> when performing our $Q$-update? ($Q$-update needs to maximize over all actions). The intuition is that neural networks can often generalize very well to unseen actions, without explicit consideration of such actions.

Consider performing policy evaluation of the behavior policy. With a typical MSE loss, we might have
$$
\begin{align*}
Q &= \min _{Q} \sum_{(s,a,s')\in \mathcal{D}}\lVert Q(s,a)-[r(s,a)+\gamma V(s')] \rVert ^{2} \\
V &=\min _{V} \sum_{(s,a)\in \mathcal{D}}\lVert V(s)-Q(s,a) \rVert ^{2}
\end{align*}
$$
which simply regresses onto the *mean* of the value of each state/action under the behavior policy. However, we can actually regress onto an *upper quantile* of these values instead, by changing our loss function.
$$
\begin{align*}
V &= \min _{V}\sum_{(s,a)\in \mathcal{D}}\ell_{\tau}(V(s)-Q(s,a))
\end{align*}
$$
where $\ell_{\tau}$ is a function that heavily penalizes $V$ underestimating $Q$, and softly penalizes $V$ overestimating $Q$. It's essentially just MSE with a multiplier on all negative values in the domain. This essentially induces our function estimator to perform implicit maximization over the actions in estimating the value function.

> [!question] Doesn't this cause erroneous overestimation? Wasn't this an issue in soft actor-critic?
> Actually, no. In SAC, overestimation was caused by optimism towards the state transitions. Here, the $Q$-function estimator still uses standard MSE to effectively regress onto the state transitions. Only the $V$ estimator now uses the modified loss function to implicitly maximize over the actions—which is precisely what we want for an RL algorithm. It would overestimate had we implemented, say, implicit maximization with just a $Q$-function. (That is, the $Q$-function's use of standard MSE ensures no erroneous overestimation).

So, implicit $Q$-learning, in practice, uses
$$
\begin{align*}
Q(s,a) &\leftarrow r(\mathbf{s},\mathbf{a})+\mathbb{E}_{\mathbf{a}'\sim \pi}[Q(\mathbf{s}',\mathbf{a}')] \\
V &\leftarrow \underset{V}{\arg\min}\ \frac{1}{N}\sum_{i=1}^{N} \ell(V(\mathbf{s}_{i})-Q(\mathbf{s}_{i},\mathbf{a}_{i})) \\
\ell &= \ell_{2}^{\tau}=\left\{ \begin{matrix}
(1-\tau)x^{2}, & x>0 \\
\tau x^{2}, & x\leq 0
\end{matrix} \right.
\end{align*}
$$
where $\tau$ is usually near $1$.

Note that it is actually possible to show that this is essentially equivalent to $Q$-learning with $Q$-updates that maximize only over actions seen in the sample data of the behavioral policy.

Thus, here's an offline-actor critic algorithm with implicit $Q$-learning.

1. Evaluate $y_{i}=r(\mathbf{s}_{i},\mathbf{a}_{i})+\gamma \hat{V}_{\psi}(\mathbf{s}_{i})$.
2. Update $\phi$ using $\nabla_{\phi}\sum_{i=1}^{B}\lVert \hat{Q}_{\phi}(\mathbf{s}_{i},\mathbf{a}_{i})-y_{i} \rVert^{2}$.
3. Update $\psi$ using $\nabla_{\psi}\sum_{i=1}^{B}\ell_{2}^{\tau}(\hat{V}_{\psi}(\mathbf{s}_{i})-\hat{Q}_{\phi}(\mathbf{s}_{i},\mathbf{a}_{i}))$.
4. $J(\theta)=\sum_{i}\log \pi_{\theta}\exp(\hat{Q}_{\phi}(\mathbf{s}_{i},\mathbf{a}_{i})-\hat{V}_{\psi}(\mathbf{s}_{i}))$.
5. $\theta\leftarrow\theta+\alpha \nabla_{\theta}J(\theta)$.

Some interesting things to note:
- No expectations under $\pi_{\theta}$.
- Actor and critic are completely independent.
## Conservative $Q$-Learning (CQL)
> [!info]
> Paper: Conservative Q-Learning for Offline Reinforcement Learning (Kumar et al.). (Yes, also Levine).

Let's now return to the idea of **pessimism** to solve distributional shift.

Our inspiration is to apply some ideas from *adversarial training*. Let's redefine our $Q$ function as
$$
\hat{Q}^{\pi}=\arg\min_{Q}\max_{\mu}\alpha \mathbb{E}_{\mathbf{s}\sim D,\mathbf{a}\sim \mu(\mathbf{a}\mid \mathbf{s})}[Q(\mathbf{s},\mathbf{a})]+\mathbb{E}_{(\mathbf{s},\mathbf{a},\mathbf{s}')\sim D}[(Q(\mathbf{s},\mathbf{a})-(r(\mathbf{s},\mathbf{a})+\mathbb{E}_{\pi}[Q(\mathbf{s}',\mathbf{a}')]))^{2}]
$$
where the first term "pushes down" on large $Q$-values, and the second term is just our regular $Q$-learning objective. ($\mu$ is our discriminator, the policy $\pi$ is our generator, in GAN terms). Note that one may show that $\hat{Q}^{\pi}\leq Q^{\pi}$ for large enough $\alpha$, or that our $\hat{Q}^{\pi}$ is a lower bound on the actual $Q$ function.

There's one issue with this solution, though—this will produce a systematic underestimate on actions close to the data. A better estimate is
$$
\begin{align*}
\hat{Q}^{\pi} =\arg\min_{Q}\max_{\mu}\ &\alpha \mathbb{E}_{\mathbf{s}\sim D,\mathbf{a}\sim \mu(\mathbf{a}\mid \mathbf{s})}[Q(\mathbf{s},\mathbf{a})]-\alpha \mathbb{E}_{(\mathbf{s},\mathbf{a})\sim D}[Q(\mathbf{s},\mathbf{a})] \\
&+\mathbb{E}_{(\mathbf{s},\mathbf{a},\mathbf{s}')\sim D}[(Q(\mathbf{s},\mathbf{a})-(r(\mathbf{s},\mathbf{a})+\mathbb{E}_{\pi}[Q(\mathbf{s}',\mathbf{a}')]))^{2}]
\end{align*}
$$
This new term essentially just pushes back up on $(\mathbf{s},\mathbf{a})$ samples in our data, and effectively cancels out with the first term. Notably, we no longer have the guarantee that $\hat{Q}^{\pi}\leq Q^{\pi}$ for *all* $(\mathbf{s},\mathbf{a})$, but we are guaranteed that $\mathbb{E}_{\pi(\mathbf{a}\mid \mathbf{s})}[\hat{Q}^{\pi}(\mathbf{s},\mathbf{a})]\leq \mathbb{E}_{\pi(\mathbf{a}\mid \mathbf{s})}[Q^{\pi}(\mathbf{s},\mathbf{a})]$ for all $\mathbf{s} \in D$.

Thus, we have a basic conservative $Q$-learning algorithm.

1. Update $\hat{Q}^{\pi}$ w.r.t. $\mathcal{L}_{\text{CQL}}(\hat{Q}^{\pi})$ using $\mathcal{D}$.
2. Update policy $\pi$, dependent on discrete/continuous action space.

We've left something out though in our discussion so far—what is this $\mu$ term? Well, it's the distribution that maximizes that inner term; however, without any regularization, it's extremely unstable over the course of training. Typically, we'll add a regularization term $\mathcal{R}(\mu)$ to the loss function, i.e.
$$
\begin{align*}
\hat{Q}^{\pi} =\arg\min_{Q}\max_{\mu}\ &\alpha \mathbb{E}_{\mathbf{s}\sim D,\mathbf{a}\sim \mu(\mathbf{a}\mid \mathbf{s})}[Q(\mathbf{s},\mathbf{a})]-\alpha \mathbb{E}_{(\mathbf{s},\mathbf{a})\sim D}[Q(\mathbf{s},\mathbf{a})]-\mathcal{R}(\mu) \\
&+\mathbb{E}_{(\mathbf{s},\mathbf{a},\mathbf{s}')\sim D}[(Q(\mathbf{s},\mathbf{a})-(r(\mathbf{s},\mathbf{a})+\mathbb{E}_{\pi}[Q(\mathbf{s}',\mathbf{a}')]))^{2}]
\end{align*}
$$
A common choice for $\mathcal{R}(\mu)$ is $\mathbb{E}_{\mathbf{s}\sim D}[\mathcal{H}(\mu(\cdot \mid \mathbf{s}))]$, or maximum entropy regularization. In this case, after learning, $\mu$ becomes proportional to $\exp(Q(\mathbf{s},\mathbf{a}))$. Notably, we can either represent $\mu$ as an actual, explicit learned policy, or we can note that
$$
\mathbb{E}_{\mathbf{a}\sim \mu(\mathbf{a}\mid \mathbf{s})}[Q(\mathbf{s},\mathbf{a})]=\log \sum_{\mathbf{a}}\exp(Q(\mathbf{s},\mathbf{a}))
$$
For discrete actions, we can calculate this quantity directly, while for continuous actions, we can use importance sampling to estimate this quantity.

> [!question] What if you didn't use a regularizer?
> If you didn't add a regularizer $\mathcal{R}(\mu)$, the maximizing $\mu$ is a distribution that assigns probability $1$ to the highest $Q$-value. Because this may change between every iteration, it is potentially unstable, and therefore slows learning. With maximum entropy regularization, the distribution is encouraged to spread out a bit more over the large $Q$-values.
## Offline-to-online RL
Now, let's turn to the problem of offline-to-online RL: using offline RL to pretrain a model, and then using online RL to fine-tune the model.

> [!warning] At the time of writing (March 2026), this is an active area of research. Take all following discussions with a grain of salt.

Okay, so what if we just run an offline RL algorithm, e.g. CQL, and then run a standard online RL algorithm starting from the same point (same policy, same $Q$-function)? Does that just work?

Unfortunately, not quite. See below.

![[offline-to-online.png|300]]

For CQL, in particular, which this graph was generated from, during the first phase of online training, the model will realize that it has drastically underestimated the $Q$-function for some previously unobserved actions, due to the pessimism of CQL. Other offline RL algorithms have similar, slightly different versions of this same problem.

The problem is the differences between the training phases.

Offline:
- Stay close to $\pi_{\beta}$.
- Conservative values (e.g. $\hat{Q}^{\pi}\leq Q^{\pi}$).
- Lots of gradient steps on the same data.

Online:
- Improve as much as possible over $\pi_{\beta}$.
- Optimistic values (e.g. $\hat{Q}^{\pi}\geq Q^{\pi}$). (important for exploration)
- Learn as quickly as possible.

There is, notably, an "embarrassingly effective" method that is not really offline RL at all, yet outperformed, for a long time, many offline RL methods in using offline data for online methods. (Efficient Online Reinforcement Learning with Offline Data, Ball et al.).

1. Initialize two buffers: online replay buffer and offline data buffer
2. Initialize value function and actor from scratch.
3. Run online RL; for every batch, sample half from offline buffer and half from replay buffer.

Obviously, this method is unsatisfying because it involves absolutely zero pretraining. Unfortunately, there isn't a clear solution to this; however, in recent years, algorithms that use *diffusion* models or *flow matching* to represent the actor have empirically produced methods with pretraining that do improve over the "embarrassingly effective" method.

It's unknown why exactly they work well, but here is one theory. In the online case, optimal $\pi_{\theta}(\mathbf{a}\mid \mathbf{s})$ is deterministic; thus, there's no need to capture some multimodal distribution for the policy. In the offline case, capturing only the best mode in the data is fine... but when capturing multiple modes, it may be easier to handle policy constraints. So, for offline-to-online, perhaps it helps to just track all the modes in the offline phase, and then focus on the best mode in the online phase.

However, it's <font color="#c0504d">hard to use diffusion as the actor</font> in RL. Optimizing the objective requires either $\nabla_{\theta}\log \pi_{\theta}(\mathbf{a}\mid \mathbf{s})$ (policy gradient) or backpropagation through the diffusion process (reparameterization). Policy gradient is inaccessible for diffusion/flow matching, and backprop is often computationally expensive and unstable (backprop through time, **BPTT**).

Let's discuss some methods that have used diffusion models, and what they did to solve the above issue.
### IDQL
Simple offline-to-online RL with diffusion model. (IDQL: Implicit Q-Learning as an actor-critic method with diffusion policies, Hansen-Estruch et al.)

1. Train $\hat{Q}_{\phi}(\mathbf{s},\mathbf{a})$ without any actor (e.g. IQL)
2. Train $\pi_{\theta}(\mathbf{a}\mid \mathbf{s})$ as a diffusion/flow model with behavioral cloning, leads to $\pi_{\theta}\approx \pi_{\beta}$.
3. At test time, sample $\{ \mathbf{a}_{1},\dots,\mathbf{a}_{K} \}$ from $\pi_{\theta}(\mathbf{a}\mid \mathbf{s})$, and pick $\arg\max_{\mathbf{a}_{k}}\hat{Q}_{\phi}(\mathbf{s},\mathbf{a}_{k})$.

The point is that no out-of-distribution actions are chosen because $\pi_{\theta}\approx \pi_{\beta}$. This is somewhat reminiscent of stochastic optimization/random shooting from [[Lecture 16#Stochastic Optimization / Random Shooting|Lecture 16]]. However, this works surprisingly well, particularly if the data produced by $\pi_{\beta}$ is somewhat decent.

### FQL
An actor that stays "close" to diffusion model. (Flow Q-Learning, Park et al.)

1. Train $\pi_{\text{flow}}(\mathbf{a}\mid \mathbf{s},\mathbf{z})$ as diffusion/flow model with behavioral cloning, where $\mathbf{z}$ is the input noise in flow matching. This leads to $\pi_{\text{flow}}\approx \pi_{\beta}$.
2. Run offline actor-critic with a special behavioral cloning regularizer, training the actor $\pi_{\theta}(\mathbf{a}\mid \mathbf{s},\mathbf{z})$ to stay close to $\pi_{\text{flow}}(\mathbf{a}\mid \mathbf{s},\mathbf{z})$, given both $\mathbf{s}$ and $\mathbf{z}$ as input. Notably, the actor is just a regular neural network, not a flow model, that is essentially *distilling* the behavior of the flow model. In particular, the objective for the actor is
$$
J(\theta) = \sum_{i}\mathbb{E}_{\mathbf{z}\sim p(\mathbf{z})}\big[\mathbb{E}_{\mathbf{a}\sim \pi_{\theta}(\mathbf{a}\mid \mathbf{s}_{i},\mathbf{z})}[\hat{Q}_{\phi}^{\pi}(\mathbf{s}_{i},\mathbf{a})]+\mathbb{E}_{\mathbf{a}\sim \pi_{\text{flow}}(\mathbf{a}\mid \mathbf{s},\mathbf{z})}[\lambda \log \pi_{\theta}(\mathbf{a}_{i}\mid \mathbf{s},\mathbf{z})]\big]
$$

### DSRL
Diffusion Steering via Reinforcement  Learning. (Steering Your Diffusion Policy with Latent Space Reinforcement Learning, Wagenmaker et al.)

Intuitively, a diffusion model actually produces an action space of *only in-distribution* actions. So, what if, instead of our actor model producing an action from the general action space, which may be out-of-distribution, we use our actor model to produce a value in the **latent space** of the diffusion model, i.e. the noise distribution, and then feed that into the diffusion model to produce an in-distribution action. That is, just run an efficient online RL algorithm in the latent space of a diffusion model!
## Model-based Offline RL
The critical concern in model-based offline RL is that we'd like to limit the impact of the distributional shift of the environment model itself, because we are unable to collect more data to correct the model's errors. (*counterfactual* questions for the model, rather than for the $Q$-function like previously).
### MOPO
> [!info] MOPO: Model-Based Offline Policy Optimization (Yu et al.)
> Also, MOReL: Model-Based Offline Reinforcement Learning (Kidambi et al.)

One solution is to simply adjust the reward function to be pessimistic about OOD states, i.e. to "*punish*" the policy for exploiting OOD states. The reward is adjusted as follows
$$
\tilde{r}(s,a)=r(s,a)-\lambda u(s,a)
$$
where $u(s,a)$ represents an **uncertainty penalty**. This can be computed by e.g. measuring disagreement between ensemble models. Subsequently, simply run any existing model-based RL method.
### COMBO
> [!info] COMBO: Conservative Offline Model-Based Policy Optimization (Yu et al.)

Alternatively, we can leverage the same ideas as in [[#Conservative $Q$-Learning (CQL)|CQL]]. Similar to how CQL "pushes down" on large $Q$-values, model-based RL can "push down" on the $Q$-values of model state-action tuples. In essence, for a model $p$, we use a $Q$-function update rule of
$$
\hat{Q}^{k+1} \leftarrow \underset{Q}{\arg\min}\ \beta(\mathbb{E}_{\mathbf{s},\mathbf{a}\sim p(\mathbf{s},\mathbf{a})}[Q(\mathbf{s},\mathbf{a})]-\mathbb{E}_{\mathbf{s},\mathbf{a}\sim \mathcal{D}}[Q(\mathbf{s},\mathbf{a})])+\frac{1}{2}\mathbb{E}_{\mathbf{s},\mathbf{a},\mathbf{s}'\sim \mathbfit{d}_{\mathbfit{f}}}[(Q(\mathbf{s},\mathbf{a})-\hat{\beta}^{\pi}\hat{Q}^{k}(\mathbf{s},\mathbf{a}))^{2}]
$$
Again, the intuition is the same. It's like a GAN: if the model, the generator, produces something that looks clearly different from real data, the $Q$-function, the discriminator, will assign low values to it.
<div style="page-break-after: always;"></div>

