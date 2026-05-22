# Lecture 11: Variational Inference
"Variational inference is an optimization-based framework for training models with latent variables."
## Latent Variable Models
Latent variables are learned as **mixture elements** that turn probabilistic models into latent variable models whose distribution is represented as a mixture of multiple distributions, i.e.
$$
\begin{align*}
p(x) &= \sum_{z}p(x\mid z)p(z) \\
p(y\mid x)&=\sum_{z}p(y\mid x,z)p(z)
\end{align*}
$$
One possible implementation of a continuous latent variable model $p(x)=\int p(x\mid z)p(z) \,\mathrm{d}z$ is the nonlinear composition of two "easy" distributions, e.g.
$$
\begin{align*}
p(x\mid z) &= \mathcal{N}(\mu_{\text{nn}(z)},\alpha_{\text{nn}(z)}) \\
p(z) &= \mathcal{N}(0,1)
\end{align*}
$$
where the $\text{nn}$ subscript implies a neural network. This is nice because it's easy to use our "easy" distributions.

Sidenote: this is applicable in RL for

- Conditional latent variable models for multi-modal policies
- Latent variable models for model-based RL (where you learn the MDP of the environment itself, more on this later in the course)

So how do we train latent variable models? Well, using maximum likelihood gives us
$$
\begin{align*}
p(x) &= \int p(x\mid z)p(z) \,\mathrm{d}z \\
\theta &\leftarrow \underset{\theta}{\arg\max}\ \frac{1}{N}\sum_{i}\log\left( \int p(x\mid z)p(z) \,\mathrm{d}z  \right)
\end{align*}
$$
This is intractable though—the maximum likelihood solution is difficult to compute because of the integral! Moreover, the integral can't be approximated with sampling because it lies inside a $\log$, and therefore <font color="#c0504d">linearity of expectation no longer applies</font> D:

(Had we tried to approximate with Monte Carlo sampling, the estimator would be *biased*).

Instead, we can use **expected log-likelihood**.
$$
\theta\leftarrow \underset{\theta}{\arg\max}\ \frac{1}{N}\sum_{i}\mathbb{E}_{z\sim p(z\mid x_{i})}[\log p_{\theta}(x_{i},z)]
$$
in essence, we "guess" the most likely value of the latent variable $z$ for $x_{i}$ by sampling from a **posterior** $p(z\mid x_{i})$ that we construct, and then pretend our guess is correct. But, we still need to compute this posterior.

This problem of calculating $p(z\mid x_{i})$ is known as **probabilistic inference**.

Sidenote: probabilistic inference is also seen in RL for

- Modeling human behavior
- Generative models + variational inference for exploration
- Reinforcement learning from human feedback (RLHF) (predict reward based on model output)
## Variational Inference
Performing probabilistic inference exactly is extremely hard; however, for practical purposes, there are *approximate* methods that work well.

First, let's pretend our posterior is extremely simple. This assumption is **incorrect**, but very convenient! (We'll see soon how we may quantify the extent to which this assumption is wrong). For instance, we can approximate $p(z\mid x_{i})\approx q_{i}=\mathcal{N}(\mu_{i},\sigma_{i})$. Critically, this approximation allows us to tractably produce a *lower bound* on $\log p(x_{i})$.

We note that
$$
\begin{align*}
\log p(x_{i}) &= \log\int p(x_{i}\mid z)p(z) \,\mathrm{d}z \\
&= \log \int p(x_{i}\mid z)p(z) \frac{q_{i}(z)}{q_{i}(z)} \,\mathrm{d}z \\ 
&= \log \mathbb{E}_{z\sim q_{i}(z)}\left[ \frac{p(x_{i}\mid z)p(z)}{q_{i}(z)} \right]
\end{align*}
$$
(Really, this is just an application of Bayes' rule).

We can then apply what is known as **Jensen's inequality**, which states that
$$
f(\mathbb{E}[y])\geq \mathbb{E}[f(y)]
$$
for any *concave function* $f$. Notably, $\log$ is a concave function. Using this, we may conclude
$$
\begin{align*}
\log p(x_{i}) &= \log \mathbb{E}_{z\sim q_{i}(z)}\left[ \frac{p(x_{i}\mid z)p(z)}{q_{i}(z)} \right] \\
&\geq  \mathbb{E}_{z\sim q_{i}(z)}\left[ \log\frac{p(x_{i}\mid z)p(z)}{q_{i}(z)} \right] \\
&\geq  \mathbb{E}_{z\sim q_{i}(z)}[\log p(x_{i}\mid z)+\log p(z)]-\mathbb{E}_{z\sim q_{i}(z)}[\log q_{i}(z)] \\
&\geq  \mathbb{E}_{z\sim q_{i}(z)}[\log p(x_{i},z)]+\mathcal{H}(q_{i})
\end{align*}
$$
where $\mathcal{H}$ is the **entropy** of the distribution. Notably, $\mathbb{E}_{z\sim q_{i}(z)}[\log p(x_{i},z)]$ is precisely our expected log-likelihood term!

Note that the definition of entropy is
$$
\mathcal{H}(p)=-\mathbb{E}_{x\sim p(x)}[\log p(x)]=-\int p(x)\log p(x) \,\mathrm{d}x 
$$
Intuitively, entropy describes...

- how random the random variable is?
- how large is the log probability in expectation under itself?

So what exactly do the two terms in the expression $\mathbb{E}_{z\sim q_{i}(z)}[\log p(x_{i},z)]+\mathcal{H}(q_{i})$ do?

1. The first term, our expected log-likelihood, induces the model to maximize probability at the *peaks* of the distribution.
2. The second term, entropy, induces the model to distribute its probability over a greater area, i.e. it encourages increased entropy in the model.

Now, note the following representation of **KL divergence**.
$$
\begin{align*}
D_{\text{KL}}(q\parallel p) &= \mathbb{E}_{x\sim q(x)}\left[ \log \frac{q(x)}{p(x)} \right] \\
&=-\mathbb{E}_{x\sim q(x)}[\log p(x)]-\mathcal{H}(q)
\end{align*}
$$
that looks suspiciously similar to our expression above...

Intuitively, KL divergence describes...

- how different are the two distributions?
- how small is the expected log probability of one distribution under another, minus entropy?

Thus, our expression $\mathbb{E}_{z\sim q_{i}(z)}[\log p(x_{i},z)]+\mathcal{H}(q_{i})$ is really very similar to KL divergence, and is a measure of the divergence between the distributions $\log p(x_{i},z)$ and $q_{i}(x)$. Notably, maximizing this expression, and therefore $\log p(x)$, is equivalent to minimizing the KL divergence, or minimizing the distance between the two distributions.

Let's now define $\mathcal{L}_{i}(p,q_{i})=\mathbb{E}_{z\sim q_{i}(z)}[\log p(x_{i},z)]+\mathcal{H}(q_{i})$. We previously alluded to how we may quantify how incorrect our initial assumption was. So, how do we measure how good a choice of $q_{i}$ is? The KL divergence $D_{\text{KL}}(q_{i}(z)\parallel p(z\mid x_{i}))$. Consider that
$$
\begin{align*}
D_{\text{KL}}(q_{i}(z)\parallel p(z\mid x_{i})) &= \mathbb{E}_{z\sim q_{i}(z)}\left[ \log \frac{q_{i}(z)}{p(z\mid x_{i})} \right] \\
&= \mathbb{E}_{z\sim q_{i}(z)}\left[ \log \frac{q_{i}(z)p(x_{i})}{p(x_{i},z)} \right] \\
&= -\mathbb{E}_{z\sim q_{i}(z)}[\log p(x_{i},z)]+\mathbb{E}_{z\sim q_{i}(z)}[\log q_{i}(z)]+\mathbb{E}_{z\sim q_{i}(z)}[\log p(x_{i})] \\
&= -\mathbb{E}_{z\sim q_{i}(z)}[\log p(x_{i},z)]-\mathcal{H}(q_{i})+\log p(x_{i}) \\
&= -\mathcal{L}_{i}(p,q_{i})+\log p(x_{i})
\end{align*}
$$
and therefore
$$
\log p(x_{i}) = D_{\text{KL}}(q_{i}(z)\parallel p(z\mid x_{i}))+\mathcal{L}_{i}(p,q_{i})
$$
In other words, $q_{i}$ produces a lower bound $\mathcal{L}_{i}(p,q_{i})\leq\log p(x_{i})$ where the error (distance between lower bound and the actual value of $\log p(x_{i})$) is precisely the KL divergence between $q_{i}(z)$ and the distribution it attempts to approximate, $p(z\mid x_{i})$.

Now, we may note that, in the expression for the KL divergence,
$$
D_{\text{KL}}(q_{i}(z)\parallel p(z\mid x_{i})) = -\mathcal{L}_{i}(p,q_{i})+\log p(x_{i})
$$
$\log p(x_{i})$ is independent of $q_{i}$. Therefore, if we maximize $\mathcal{L}_{i}(p,q_{i})$ with respect to $q_{i}$, we minimize KL divergence! Thus,

- Maximizing $\mathcal{L}$ with respect to $p$ increases the log-likelihood
- Maximizing $\mathcal{L}$ with respect to $q_{i}$ tightens the lower bound

Thus, we can actually select $q_{i}$ intelligently by iteratively maximizing $\mathcal{L}_{i}$ w.r.t $q_{i}$.

Here's the final idea we derive from the above deductions.

1. For each $x_{i}$ or mini-batch:
	
	1. Sample $z\sim q_{i}(z)$ and approximating $\nabla_{\theta}\mathcal{L}_{i}(p,q_{i})\approx \nabla_{\theta}\log p_{\theta}(x_{i}\mid z)$.
	2. $\theta\leftarrow\theta+\alpha \nabla_{\theta}\mathcal{L}_{i}(p,q_{i})$.
	3. Update $q_{i}$ to maximize $\mathcal{L}_{i}(p,q_{i})$.

Where the optimization step w.r.t $q_{i}$ may be done with gradient ascent or another optimization method. For instance, assuming $q_{i}(z)=\mathcal{N}(\mu_{i},\sigma_{i})$, we use the gradients $\nabla_{\mu_{i}}\mathcal{L}_{i}(p,q_{i})$ and $\nabla_{\sigma_{i}}\mathcal{L}_{i}(p,q_{i})$ to perform gradient ascent on $\mu_{i}$ and $\sigma_{i}$.

There's just one problem... <font color="#c0504d">there are way too many parameters</font>! In fact, it scales with the size of the dataset, i.e. $\lvert \theta \rvert+(\lvert \mu_{i} \rvert+\lvert \sigma_{i} \rvert )\times N$. We'll address this issue next lecture; in particular, how deep learning makes this tractable :)

> [!tip]
> The Expectation-Maximization algorithm is actually a special case of variational inference! The $E$ step corresponds to updating $q_{i}$, and the $M$ step corresponds to updating $\theta$.

> [!tip]
> The lower bound we derived is known as the [Evidence Lower Bound](https://en.wikipedia.org/wiki/Evidence_lower_bound) or **ELBO**.
<div style="page-break-after: always;"></div>

