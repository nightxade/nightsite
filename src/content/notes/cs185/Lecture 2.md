# Lecture 2
**Markov Property**
In RL problems, *states* follow the Markov property, i.e. the future state depends only on the current state, not the previous states that led up to it. In other words, the state is *memoryless*.

**Partially observed RL**
Actions $a_{t}$, states $s_{t}$, observations $o_{t}$.
$$
\begin{align*}
& \pi_{\theta}(\mathbf{a}_{t}\mid \mathbf{o}_{t}) \\
& p(\mathbf{o}_{t}\mid \mathbf{s}_{t}) \\
& p(\mathbf{s}_{t+1}\mid \mathbf{s}_{t},\mathbf{a}_{t})
\end{align*}
$$
**Fully observed RL**
$\mathbf{o}_{t}=\mathbf{s}_{t}$, so observations are unnecessary.
$$
\begin{align*}
& \pi_{\theta}(\mathbf{a}_{t}\mid \mathbf{s}_{t}) \\
& p(\mathbf{s}_{t+1}\mid \mathbf{s}_{t},\mathbf{a}_{t})
\end{align*}
$$
**Behavioral Cloning**
What if we wanted to do supervised learning? When collecting data, we can measure several *demonstration trajectories*
$$
\{ (\mathbf{o}_{1}^{(i)},\mathbf{a}_{1}^{(i)},\dots ,\mathbf{o}_{H}^{(i)},\mathbf{a}_{H}^{(i)}) \}_{i=1}^{N}
$$
Behavioral cloning, a form of **imitation learning**, involves maximizing the log-probability of the policy—this is effectively just supervised learning/maximum likelihood estimation with a neural network.
$$
\underset{ \theta }{ \arg\max  }\sum_{i=1}^{N} \sum_{t=1}^{H} \log \pi_{\theta}(\mathbf{a}_{t}^{(i)},\mathbf{o}_{t}^{(i)})
$$
The algorithm itself would take the raw data (e.g. image) and process it with an encoder to produce the *action distribution parameters*. For discrete actions, these parameters are logits that are then processed with a softmax. For continuous actions, these parameters are actual distribution parameters, i.e. they are values like Gaussian means and covariance that parametrize the distributions of each action.
$$
\begin{align*}
& \text{Output: } \begin{bmatrix}
f_{1}(\mathbf{o}_{t}) \\
\vdots  \\
f_{A}(\mathbf{o}_{t})
\end{bmatrix}
&& \implies && p(\mathbf{a}_{t}=1\mid \mathbf{o}_{t})= \frac{\text{exp}(f_{1}(\mathbf{o}_{t}))}{\sum_{i=1}^{A} \exp(f_{i}(\mathbf{o}_{t}))} \\

& \text{Output: } \begin{bmatrix}
\mu(\mathbf{o}_{t}) \\
\Sigma(\mathbf{o}_{t}) \\
\end{bmatrix}
&& \implies && p(\mathbf{a}_{t}\mid \mathbf{o}_{t})=\mathcal{N}(\mathbf{a}\mid \mu(\mathbf{o}_{t}),\Sigma(\mathbf{o}_{t}))
\end{align*}
$$

**Does Behavioral Cloning Work?**
The main issue with behavioral cloning is that, in RL, data is **not i.i.d.**, i.e. earlier actions in the sequence affect later states. However, after a slight divergence from the training trajectory (due to a small error from the model), the error will be *amplified* across the future trajectory; as the model diverges further from the training trajectory, its error subsequently grows. This problem may be addressed in several ways:

- Change the algorithm (DAgger)
- Use better models that make fewer mistakes
- Smarter data collection/augmentation
- Multi-task learning

But first, we formalize the notion of this divergence, which is termed **distributional shift**, i.e. $p_{\text{train}}(\mathbf{x})\neq p_{\text{test}}(\mathbf{x})$. At some point in testing, $p_{\pi_{\theta}}(\mathbf{o}_{t})\neq p_{\text{data}}(\mathbf{o}_{t})$. We'll now try and quantify just how bad this distributional shift can be for behavioral cloning. Note that this theoretical view is *not* necessarily representative of real-world, typical behavior.

Assume the data is produced by a good policy $\pi^{*}(\mathbf{s}_{t})$. (Note that we are using the observed case for simplicity). We define a cost function
$$
c(\mathbf{s}_{t},\mathbf{a}_{t})=\left\{ \begin{matrix}
0, & \mathbf{a}_{t}=\pi^{*}(\mathbf{s}_{t}) \\
1,& \text{otherwise}
\end{matrix} \right.
$$
The expected value of our cost at time step $t$ w.r.t. actions sampled from $\pi_{\theta}$ is thus
$$
\mathbb{E}_{\mathbf{a}_{t}\sim \pi_{\theta}(\mathbf{a}_{t}\mid \mathbf{s}_{t})} = \sum_{\mathbf{a}_{t}}\pi_{0}(\mathbf{a}_{t}\mid \mathbf{s}_{t})c(\mathbf{s}_{t},\mathbf{a}_{t})=\pi_{0}(\mathbf{a}_{t}\neq \pi^{*}(\mathbf{s}_{t})\mid \mathbf{s}_{t})
$$
In other words, this represents the probability of mistakes. Then, the expected total number of mistakes is
$$
\mathbb{E}[M]=\sum_{t=1}^{H} \mathbb{E}_{\mathbf{a}_{t}\sim \pi_{\theta}(\mathbf{a}_{t}\mid \mathbf{s}_{t}),\mathbf{s}_{t}\sim p_{\pi_{\theta}}(\mathbf{s}_{t})}[c(\mathbf{s}_{t},\mathbf{a}_{t})]
$$
Note that $\mathbf{s}_{t}\sim p_{\pi_{\theta}}(\mathbf{s}_{t})$ represents the distribution of possible states $\mathbf{s}_{t}$ at time $t$ for the policy $\pi_{\theta}$, and it *depends* on past actions.

 Let's assume $\pi_{\theta}(\mathbf{a}\neq \pi^{*}(\mathbf{s})\mid \mathbf{s})\leq\epsilon$, $\forall \mathbf{s}\in \mathcal{D}_{\text{train}}$. In other words, the probability of making a mistake is $\leq\epsilon$ for each state in the dataset. Then, 
 $$
\begin{align*}
\mathbb{E}[M] &\leq \epsilon H+(1-\epsilon)(\epsilon(H-1)+(1-\epsilon)(\dots )) \\
&\leq  \epsilon H^{2}
\end{align*}
$$
In other words, the expected number of mistakes increases *quadratically* in the horizon. This assumption, however, does not generalize to most datasets.

Instead, we can make a more realistic assumption, in which, for any state sampled from the training distribution (not just the states observed in the training dataset), we are unlikely to make a mistake, i.e.
$$
\mathbb{E}_{\mathbf{s}_{t}\sim p_{\text{train}}(\mathbf{s}_{t})}[\pi_{\theta}(\mathbf{a}_{t}\neq \pi^{*}(\mathbf{s}_{t})\mid \mathbf{s}_{t})] \leq  \epsilon
$$
Then, we can note the probability of some state $\mathbf{s}_{t}$ as
$$
p_{\pi_{\theta}}(\mathbf{s}_{t}) = (1-\epsilon)^{t}p_{\text{train}}(\mathbf{s}_{t}) + (1-(1-\epsilon)^{t})p_{\text{mistake}}(\mathbf{s}_{t})
$$
We may subsequently calculate **total variation divergence (TVD)**, a measure of distance between two distributions similar to *KL divergence*
$$
D_{\text{TV}}(p,q)=\frac{1}{2}\sum_{x}\lvert p(x)-q(x) \rvert \leq 1
$$
Through some simplifications, we note that
$$
\begin{align*}
D_{\text{TV}}(p_{\text{train}},p_{\pi_{\theta}}) &= \frac{1}{2}\sum_{\mathbf{s}_{t}}\lvert p_{\text{train}}(\mathbf{s}_{t})-p_{\pi_{\theta}}(\mathbf{s}_{t}) \rvert \\
&= \frac{1}{2}\sum_{\mathbf{s}_{t}}\lvert p_{\text{train}}(\mathbf{s}_{t})-(1-\epsilon)^{t}p_{\text{train}}(\mathbf{s}_{t}) + (1-(1-\epsilon)^{t})p_{\text{mistake}}(\mathbf{s}_{t}) \rvert \\
&= (1-(1-\epsilon)^{t}) \frac{1}{2}\sum_{\mathbf{s}_{t}}\lvert p_{\text{train}}(\mathbf{s}_{t})-p_{\text{mistake}}(\mathbf{s}_{t}) \rvert 
\end{align*}
$$
From the upper-bound on TVD $\leq1$ and the identity $(1-\epsilon)^{t}\geq 1-\epsilon t$ for $\epsilon \in[0,1]$, we find
$$
\begin{align*}
D_{\text{TV}}(p_{\text{train}},p_{\pi_{\theta}}) &\leq  (1-(1-\epsilon)^{t}) \\
&\le \epsilon t
\end{align*}
$$
In other words, the distance between probability distributions increases linearly in $t$. This bound allows us to derive, after several simplifications, that
$$
E[M]\leq \sum_{t=1}^{H} \epsilon+2\epsilon t
$$
or that $E[M]\in O(\epsilon H^{2})$. Thus, the error increases *quadratically* with the horizon. <p style="margin: 0px 0px 0px 0px;" align="right"> $\square$ </p>
This analysis, however, is *pessimistic*, as it assumes we cannot recover from mistakes. Paradoxically, then, this implies that imitation learning can achieve better performance if the data includes more mistakes and subsequent recoveries.

**Solution 1: DAgger:** **D**ataset **A**ggregation
We'd like to make $p_{\text{data}}(\mathbf{o}_{t})\approx p_{\pi_{\theta}}(\mathbf{o}_{t})$, i.e. reduce distributional shift. DAgger focuses on being smart about $p_{\text{data}}(\mathbf{o}_{t})$ for this purpose—its goal is to effectively collect training data from $p_{\pi_{\theta}}(\mathbf{o}_{t})$ instead of $p_{\text{data}}(\mathbf{o}_{t})$!

1. Train $\pi_{\theta}(\mathbf{a}_{t}\mid \mathbf{o}_{t})$ from human data $\mathcal{D}$.
2. Run $\pi_{\theta}(\mathbf{a}_{t}\mid \mathbf{o}_{t})$ to produce the model dataset $\mathcal{D}_{\pi}$.
3. Ask a human to label $D_{\pi}$ with actions $\mathbf{a}_{t}$.
4. Aggregate $D\leftarrow D\cup D_{\pi}$.
5. Repeat.

This iteratively aligns the training dataset to further match the states/observations that the policy $\pi_{\theta}$ experiences.

The main difficulty with DAgger, however, is the production of labels by humans. This is not only expensive, but humans may not even be able to classify each observation, depending on the task.

A common variant of DAgger aimed to fix this issue is to replace human labeling with human intervention.

1. Train $\pi_{\theta}(\mathbf{a}_{t}\mid \mathbf{o}_{t})$ from human data $\mathcal{D}$.
2. Run $\pi_{\theta}(\mathbf{a}_{t}\mid \mathbf{o}_{t})$.
3. Ask a human to *intervene* or take over at some time step $t$.
4. Add all $(\mathbf{o}_{t},\mathbf{u}_{t})$ examples to the human data $\mathcal{D}$.
5. Repeat.
<div style="page-break-after: always;"></div>

