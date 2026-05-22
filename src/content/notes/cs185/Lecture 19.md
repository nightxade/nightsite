# Lecture 19: Exploration
## The Exploration Problem
The problem lies in the fact that the intended objective may *correlate weakly* with rewarding events, depending on the reward function design. This issue is particularly exacerbated for *temporally extended* tasks, in which the model, before discovering any trajectory that achieves the objective, may be stuck stumbling through seemingly random or odd trajectories for a while.

Ultimately, the exploration problem poses the following (essentially equivalent) questions:
- How can an agent discover high-reward strategies that require a temporally extended sequence of complex behaviors that are individually not rewarding?
- How can an agent decide whether to attempt new behaviors (**exploration**) or continue doing the best thing it knows so far (**exploitation**)?

That begs the question—can we derive an optimal exploration strategy? And, for that purpose, what does "optimal exploration" even mean?
### The Tractability of Exploration
RL problems lie on a spectrum of *theoretical tractability* with regards to the derivation of an optimal exploration strategy. Thus, in practice, an optimal strategy is derived on a much simpler task (e.g. multi-arm bandits) and heuristically applied to a more complex task (e.g. modern deep RL problems).

For the easiest situations, multi-arm bandits and contextual bandits, we can frame exploration as *POMDP identification*. For small, tabular MDPs, we can frame exploration as *Bayesian model identification*. For large or infinite MDPs, optimal methods are essentially impossible to find/apply, so we take inspiration from simpler problems.
### Bandits
See [[2. Multi-armed Bandits|these notes]] from [[Reinforcement Learning, An Introduction, 2nd Edition.pdf|Reinforcement Learning: An Introduction]] by Barton and Sutton.

Note that, because the bandits problems forms a POMDP with the observations as the rewards and the states as the latent reward distribution of each bandit, it's possible to simply run policy gradient to find our optimal policy. This is a bit overkill, though :)
### Measurement of Exploration Optimality
We can measure how good an exploration algorithm with **regret**: the difference in cumulative reward from the optimal policy at time step $H$.
$$
\text{Reg}(H)=H\mathbb{E}[r(a^{*})]-\sum_{t=1}^{H} r(a_{t})
$$
### Optimistic Exploration with Bandits
Let $\hat{\mu}_{a}$ be the average reward for each action $a$. An exploitative strategy would involve *greedily* choosing $a=\arg\max\hat{\mu}_{a}$. However, this precludes exploration. Instead, by using an *optimistic estimate* of an action's value, i.e. $a=\arg\max\hat{\mu}_{a}+C\sigma_{a}$, where $\sigma_{a}$ is some sort of variance estimate/measure of uncertainty. A common choice of that second term is $\sqrt{ \frac{2\ln H}{N(a)} }$, where $N$ is the number of times $a$ has been picked so far. This is the same as [[2. Multi-armed Bandits#2.7 Upper-Confidence-Bound Action Selection|UCB action selection]] from the RL book notes. With this choice, $\text{Reg}(H)\in O(\log H)$, which is provably optimal!
## Optimistic Exploration in Deep RL
As in UCB, we'd like to apply some sort of "exploration bonus" to actions less taken/actions we are more uncertain about.

For MDPs, we can use a similar **count-based exploration** method, i.e. measuring uncertainty inversely with $N(\mathbf{s},\mathbf{a})$ or $N(\mathbf{s})$, a measure of visitations to that state-action pair/state. We can modify our reward function to be
$$
r^{+}(\mathbf{s},\mathbf{a})=r(\mathbf{s},\mathbf{a})+\mathcal{B}(N(\mathbf{s}))
$$
where $\mathcal{B}$ is some function inversely proportional to $N(\mathbf{s})$. So, we can just augment any model-free RL algorithm with this reward function to promote exploration... right?

Unfortunately, the issue with this count-based exploration method for deep RL problems is inherently the *massive* state/action space. For instance, consider an RL problem that uses images as state; literally counting the number of visits to each image is... probably not smart considering the number of possible images. It gets even worse if you consider a continuous task with states represented by floating point numbers! However, we know that some states are more similar than others... can we use this notion of "*familiarity*" to promote exploration of only states for which we have not seen similar states?

Well, what if we fit a **generative model** to our replay buffer of observed states? Then, we can directly quantify the familiarity of a state by simply evaluating the probability density model on it.

However, we'd also like to convert the probability density of a state to a "*pseudo-count*." We can note that the true probability density is represented by
$$
P(s)=\frac{N(\mathbf{s})}{n}
$$
where $N(s)$ is the number of visits to state $s$ and $n$ is the total number of states visited. Then, after observing a visit to $\mathbf{s}$, the probability becomes
$$
P'(\mathbf{s})=\frac{N(\mathbf{s})+1}{n+1}
$$
So we'd like our density model to obey these rules, because then we can use the probability density estimates to solve for the counts.

The paper Unifying Count-Based Exploration by Bellemare et al. describes a solution.
1. Fit model $p_{\theta}(\mathbf{s})$ to $\mathcal{D}$.
2. Take step $i$ and observe $\mathbf{s}$.
3. Fit new model $p_{\theta'}(\mathbf{s})$ to $\mathcal{D}\cap \{ \mathbf{s}_{i} \}$.
4. Use $p_{\theta}(\mathbf{s}_{i})$ and $p_{\theta'}(\mathbf{s}_{i})$ to estimate $\hat{N}(\mathbf{s})$.
5. Set $r_{i}^{+}=r_{i}+\mathcal{B}(\hat{N}(s))$.

Where we estimate $\hat{N}(\mathbf{s})$ by solving the system of equations
$$
p_{\theta}(\mathbf{s}_{i})=\frac{\hat{N}(\mathbf{s}_{i})}{\hat{n}}\qquad p_{\theta'}(\mathbf{s}_{i})=\frac{\hat{N}(\mathbf{s}_{i})+1}{\hat{n}+1}
$$
with unknown $\hat{N}(\mathbf{s}_{i})$ and $\hat{n}$ turning out to be
$$
\hat{N}(\mathbf{s}_{i})=\hat{n}p_{\theta}(\mathbf{s}_{i})\qquad \hat{n}=\frac{1-p_{\theta'}(\mathbf{s}_{i})}{p_{\theta'}(\mathbf{s}_{i})-p_{\theta}(\mathbf{s}_{i})}p_{\theta}(\mathbf{s}_{i})
$$

One final consideration: what bonus function $\mathcal{B}$ to use? There are a lot that work fine, just make sure it's *inversely proportional* to $N(s)$. Here are a couple examples from literature.
- UCB: $\mathcal{B}(N)=\sqrt{ \frac{2\ln n}{N} }$.
- MBIE-EB (Strehl & Littman): $\mathcal{B}(N)=\sqrt{ \frac{1}{N(s)} }$.
- BEB (Kolter & Ng): $\mathcal{B}(N)=\frac{1}{N(s)}$.
### Heuristic Estimation of Counts via Errors
Using a generative model $p_{\theta}(\mathbf{s})$ is great, but, in practice, you may not even need a complex model. In reality, we primarily require $p_{\theta}(\mathbf{s})$ to, not necessarily be a true density model, but to simply be large for familiar states and small for unfamiliar states. We can develop some simple heuristics for this purpose!

Consider a target function $f^{*}(\mathbf{s},\mathbf{a})$. Given a buffer $\mathcal{D}=\{ (\mathbf{s}_{i},\mathbf{a}_{i}) \}$, attempt to fit a function $\hat{f}_{\theta}(\mathbf{s},\mathbf{a})$ to model $f^{*}(\mathbf{s},\mathbf{a})$. In particular, we expect that $\hat{f}_{\theta}$, to some extent, will **overfit** to the dataset, producing near-zero error for any $\mathbf{s},\mathbf{a}\sim \mathcal{D}$ and some non-negligible error for unseen states. Thus, we can actually use the error of our model, e.g. $\mathcal{E}(\mathbf{s},\mathbf{a})=\lVert \hat{f}_{\theta}(\mathbf{s},\mathbf{a})-f^{*}(\mathbf{s},\mathbf{a}) \rVert^{2}$ as a bonus instead!

There are a number of choices for $f^{*}(\mathbf{s},\mathbf{a})$. One common choice is $f^{*}(\mathbf{s},\mathbf{a})=\mathbf{s}'$, i.e. next-state prediction. An even simpler choice is to literally just make $f^{*}$ a neural network initialized with completely random parameters. The latter choice is known as **RND** (Random Network Distillation, Burda et al.) and is the most widely used deep RL exploration method used today.
<div style="page-break-after: always;"></div>

