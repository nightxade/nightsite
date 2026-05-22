# Lecture 24: Multi-Task and Hierarchical RL
Machine learning as a field is increasingly moving away from specialist models, whose aims are to tackle specific individual tasks, towards generalist foundation models trained on large, diverse dataset that are subsequently fine-tuned for a desired purpose.

The equivalent of generalist foundation models in reinforcement learning is **multi-task RL**. In standard RL, we defined our objective as
$$
\theta\leftarrow \underset{\theta}{\arg\max}\ \mathbb{E}_{\tau \sim \pi_{\theta}(\tau)}[r(\tau)]
$$
while in multi-task RL, we define our objective as
$$
\theta\leftarrow \underset{\theta}{\arg\max}\ \mathbb{E}_{\omega \sim p(\omega)}[\mathbb{E}_{\tau \sim \pi_{\theta}(\tau \mid\omega)}[r_{\omega}(\tau)]]
$$
where $\tau$ represents the task being accomplished, and is often denoted the **task context**. Note also that
$$
\pi_{\theta}(\tau \mid\omega) = p(\mathbf{s}_{1}) \prod_{t=1}^{H} \pi_{\theta}(\mathbf{a}_{t}\mid \mathbf{s}_{t},\omega)p(\mathbf{s}_{t+1}\mid \mathbf{s}_{t},\mathbf{a}_{t})
$$
which is equivalent to standard RL except the policy's choice of actions, of course, additionally depends on the task context.
## Why Multi-Task RL?
But, why even do multi-task RL? What's the point?

One reason is **efficiency**. Perhaps we want to train robots that can operate in the real-world and individually perform different tasks. Intuitively, training a model for each task results in some redundancies in learning—there is some subset of model knowledge that comprises knowledge that is useful for all models' objectives (e.g., learning how to grab and move an object, "understanding" the laws of physics). Thus, if we instead train a single, multi-task RL model on a larger dataset of general trajectories from the real-world, we can produce a generalist foundation model capable of real-world interaction. Then, through fine-tuning (or, even just directly using the multi-task RL model), we can produce a model that does a specific task.

Another reason is **transfer learning**. Some problems may be very difficult for naive RL algorithms to effectively learn to solve for a variety of reasons—high-dimensional state/action spaces, sparse rewards, etc. Thus, it may be helpful to train a model on prior, much easier tasks that contribute useful knowledge for solving the difficult task.
## Multi-task RL Fundamentals
First, an key question: is multi-task RL *any different* from regular RL?

The answer is... *no*! Multi-task RL corresponds precisely to single-task RL in a joint MDP, where the MDP samples the task context, $\omega$, on the first time step, and continues as usual thereafter. Of course, however, different types of RL algorithms may be better suited for the unique MDP structure of multi-task RL problems.

In fact, multi-task RL is comparatively difficult for standard RL algorithms. Schaul et al. discuss this in their paper *Ray Interference: a Source of Plateaus in Deep Reinforcement Learning*. They note that, when a $Q$-learning model achieves sufficient performance on an easier task, the model tends to narrow the policy and prioritize exploiting the reward of that easier task/making progress on this single task, rather than exploring the other tasks that they have little progress in. For intuition, consider how a unimodal Gaussian policy gradient would act given this multi-task problem. Thus, while we'd like for these easier tasks to provide a [*curriculum*](https://en.wikipedia.org/wiki/Curriculum_learning) for the harder tasks, these easy tasks can instead *interfere* with the learning of harder tasks for this reason. Notably, the same problem does *not* appear in supervised (imitation) learning!

How do we reduce this issue, then, and allow the policy to learn in all tasks? One potential idea is a curriculum learning solution—**relabeling**, or reusing past experience from one context in a different context. For instance, a robot's trajectory for folding laundry can be reused as a trajectory for folding origami, except with the trajectory relabeled according to the reward function of the origami folding task. However, one may notice that this solution is inapplicable to all multi-task RL problems; in particular, this is only sensible if the different tasks share a state and action space. (If the model is presented two tasks that involve playing two completely different games, relabeling is likely unsuitable).
## Goal-conditioned RL
A special kind of multi-task RL is known as **goal-conditioned RL**, where we condition our policy on a *goal state* (notably, an actual state, not an arbitrary task), i.e. $\pi(\mathbf{a}\mid \mathbf{s},\mathbf{g})$. Then, we define our reward function by, e.g., the distance of the current state from the goal state or $r(\mathbf{s}, \mathbf{g})=\delta(\lVert \mathbf{s}-\mathbf{g} \rVert\leq\epsilon)$.

This is convenient since there's no need to manually design rewards for each task, only a reward function parameterized by the goal state. However, it's often harder to train a model for goal-conditioned RL, and not all tasks can be framed as a goal-conditioned RL problem.

Nonetheless, there are some tricks we can apply to make goal-conditioned RL easier. In fact, we can apply the *relabeling* trick to every trajectory in our dataset/any trajectory the policy produces. For any trajectory that ended at a state $\mathbf{s}_{H}$, we relabel the trajectory's reward using the reward function with $\mathbf{g}=\mathbf{s}_{H}$, and thus this sample becomes an example of a successful trajectory for this goal.

This idea is particularly powerful when our desired goal state, $\mathbf{g}^{*}$, is extremely difficult to reach, likely due to sparse rewards or otherwise complex environments. The key idea is that trajectories that fail for our desired goal $\mathbf{g}^{*}$ still succeed for another goal (the end state), and the model can gain some knowledge from this trajectory's successful navigation to this other goal.

Note that, in an actual algorithm, we will usually choose (with some random probability) between using $\mathbf{g}=\mathbf{s}_{H}$ and $\mathbf{g}=\mathbf{g}^{*}$ for each sample. The reason is that using $\mathbf{g}=\mathbf{s}_{H}$ introduces **hindsight bias**, bias induced by the selection of $\mathbf{g}$ based on the end state of the trajectory. This is naturally an overly optimistic view of the sample. Another method of reducing hindsight bias is to sample a state according to, e.g., a geometric distribution over the time steps, i.e. change the probability of a state being sampled depending on the proximity to the goal state in time.

Additionally, using the relabeling trick forces us to use an *off-policy* RL algorithm. This is because, though we relabel the rewards of the trajectory, the current policy, $\pi_{\theta}$, may not have ever taken those actions or produced this trajectory!
### Distances
Now, consider changing the reward function to be $r(\mathbf{s},\mathbf{g})=-\delta(\mathbf{s}\neq \mathbf{g})$ with a discounting factor of $\gamma=1$. Then, the value function can be written as
$$
V(\mathbf{s},\mathbf{g}) = -\delta(\mathbf{s}\neq \mathbf{g}) + \mathbb{E}_{\mathbf{s}'}[V(\mathbf{s}',\mathbf{g})]\delta(\mathbf{s}\neq \mathbf{g})
$$
where the last $\delta(\mathbf{s}\neq \mathbf{g})$ is there so that bootstrapping does not occur when the goal is reached. Notably, $V(\mathbf{s},\mathbf{g})$ then behaves somewhat like a *distance function* (it is a quasimetric of distance), representing the number of time steps between $\mathbf{s}$ and $\mathbf{g}$. Notably, this does satisfy the **triangle inequality**, i.e.
$$
V(\mathbf{s},\mathbf{g}) \leq  V(\mathbf{s},\mathbf{w})+V(\mathbf{w},\mathbf{g})
$$
This can be utilized for **planning** with goal-conditioned value functions. For instance, a model can use several short-horizon trajectories to then plan a long-horizon trajectories along a sequence of waypoints, where the distance between waypoints is optimized leveraging the triangle inequality property of the value function. It can also be used as an **auxiliary loss function**.
## Successor Representations
> Claim: goal-conditioned RL is a lot like a model.

Consider the quasi-distance value function. This is effectively a model that predicts how easy it is to reach various states! Let's formalize this notion a bit.

First, what kind of model do we need to even evaluate a policy? Consider that
$$
\begin{align*}
V^{\pi}(\mathbf{s}_{t}) &= \sum_{t=t'}^{\infty} \gamma^{t'-t}\mathbb{E}_{p(\mathbf{s}_{t'}\mid \mathbf{s}_{t})}[r(\mathbf{s}_{t'})] \\
&= \sum_{\mathbf{s}} \underbrace{ \left(\sum_{t=t'}^{\infty} \gamma^{t'-t}p(\mathbf{s}_{t'}=\mathbf{s}\mid \mathbf{s}_{t})\right) }_{ p_{\pi}(\mathbf{s}_{\text{future}}=\mathbf{s}\mid \mathbf{s}_{t}) }r(\mathbf{s})
\end{align*}
$$
We can make $p_{\pi}(\mathbf{s}_{\text{future}}=\mathbf{s}\mid \mathbf{s}_{t})$ into a probability distribution by normalizing
$$
\begin{align*}
p_{\pi}(\mathbf{s}_{\text{future}}=\mathbf{s}\mid \mathbf{s}_{t}) = (1-\gamma)\sum_{t'=t}^{\infty} \gamma^{t'-t}p(\mathbf{s}_{t'}=\mathbf{s}\mid \mathbf{s}_{t})
\end{align*}
$$
And thus we can write
$$
\begin{align*}
V^{\pi}(\mathbf{s}_{t}) &= \frac{1}{1-\gamma}\underbrace{ \sum_{\mathbf{s}}p_{\pi}(\mathbf{s}_{\text{future}}=\mathbf{s}\mid \mathbf{s}_{t})r(\mathbf{s}) }_{ \mu^{\pi}(\mathbf{s}_{t})^{\top}\vec{r} }
\end{align*}
$$
where $\mu_{i}^{\pi}(\mathbf{s}_{t}):=p_{\pi}(\mathbf{s}_{\text{future}}=i\mid \mathbf{s}_{t})$. This is known as the **successor representation**.

We can write a Bellman-like backup expression for $\mu$.
$$
\begin{align*}
\mu_{i}^{\pi}(\mathbf{s}_{t}) = (1-\gamma)\delta(\mathbf{s}_{t}=i)+\gamma \mathbb{E}_{\mathbf{a}_{t}\sim \pi(\mathbf{a}_{t}\mid \mathbf{s}_{t}),\mathbf{s}_{t+1}\sim p(\mathbf{s}_{t+1}\mid \mathbf{s}_{t},\mathbf{a}_{t})}[\mu_{i}^{\pi}(\mathbf{s}_{t+1})]
\end{align*}
$$
It's not clear how useful success representations are yet, though—in fact, they seem much more complicated and difficult to learn and use than value functions, and their $\lvert \mathcal{S} \rvert$ vector forms seem inapplicable to near-infinite state spaces...
### Successor Features
We noted that (ignoring the normalization factor)
$$
V^{\pi}(\mathbf{s}_{t}) = \mu^{\pi}(\mathbf{s}_{t})^{\top}\vec{r}
$$
Or that the value of a state is equivalent to the dot product between the successor representations and the reward vector for all states. We can generalize this notion by replacing the reward with any other property or quality of states (anything else that is a function of a state), i.e.
$$
\psi_{j}^{\pi}(\mathbf{s}_{t}) = \mu^{\pi}(\mathbf{s}_{t})^{\top}\vec{\phi}_{j}
$$
Now, if we choose our **successor features** $\phi$ appropriately such that
$$
\begin{align*}
r(\mathbf{s}) = \sum_{j}\phi_{j}(\mathbf{s})w_{j} = \phi(\mathbf{s})^{\top}\mathbf{w}
\end{align*}
$$
for some weights $\mathbf{w}$, we may show that
$$
\begin{align*}
V^{\pi}(\mathbf{s}_{t}) &= \mu^{\pi}(\mathbf{s}_{t})^{\top}\vec{r} \\
&= \mu^{\pi}(\mathbf{s}_{t})^{\top}\sum_{j}\vec{\phi}_{j}\mathbf{w} \\
&= \sum_{j}\mu^{\pi}(\mathbf{s}_{t})^{\top}\vec{\phi}_{j}\mathbf{w} \\
&= \sum_{j}\psi_{j}^{\pi}(\mathbf{s}_{t})w_{j} \\
&= \psi^{\pi}(\mathbf{s}_{t})^{\top}\mathbf{w}
\end{align*}
$$
In other words, the value function may be represented with, rather than vectors the size of the state space, vectors that are the size of the successor features vectors! Provided, of course, the features are sufficient to full represent the reward function.

So, why is this potentially better than evaluating a policy by learning a value function? The key idea is that, if the reward function can be represented with a set of features with size $\ll$ the number of states, learning the successor representation is much easier, while still evaluating the policy!

Note that the Bellman-like backup becomes
$$
\psi_{j}^{\pi}(\mathbf{s}_{t}) = \phi_{j}(\mathbf{s}_{t}) + \gamma \mathbb{E}_{\mathbf{a}_{t}\sim \pi(\mathbf{a}_{t}\mid \mathbf{s}_{t}),\mathbf{s}_{t+1}\sim p(\mathbf{s}_{t+1}\mid \mathbf{s}_{t},\mathbf{a}_{t})}[\psi_{j}^{\pi}(\mathbf{s}_{t+1})]
$$
and that it's also possible to construct a $Q$-function-like version.
$$
\psi_{j}^{\pi}(\mathbf{s}_{t},\mathbf{a}_{t}) = \phi_{j}(\mathbf{s}_{t}) + \gamma \mathbb{E}_{\mathbf{s}_{t+1}\sim p(\mathbf{s}_{t+1}\mid \mathbf{s}_{t},\mathbf{a}_{t}),\mathbf{a}_{t+1}\sim \pi(\mathbf{a}_{t+1}\mid \mathbf{s}_{t+1})}[\psi_{j}^{\pi}(\mathbf{s}_{t+1},\mathbf{a}_{t+1})]
$$
and then the $Q$-function itself would just be
$$
Q^{\pi}(\mathbf{s}_{t},\mathbf{a}_{t}) \approx \psi^{\pi}(\mathbf{s}_{t},\mathbf{a}_{t})^{\top}\mathbf{w}
$$
when $r(\mathbf{s}_{t})\approx \phi(\mathbf{s}_{t})^{\top}\mathbf{w}$.
### Using Successor Features
Now, in what ways can we apply successor features?

**Idea 1**: recover a $Q$-function very quickly.

1. Train $\psi^{\pi}(\mathbf{s}_{t},\mathbf{a}_{t})$ via Bellman backups with a standard RL algorithm.
2. Get some reward samples $\{ \mathbf{s}_{i},r_{i} \}$.
3. Optimize $\mathbf{w}\leftarrow \arg\min_{\mathbf{w}}\sum_{i}\lVert \phi(\mathbf{s}_{i})^{\top}\mathbf{w}-r_{i} \rVert^{2}$.
4. Recover $Q^{\pi}(\mathbf{s}_{t},\mathbf{a}_{t})\approx \psi^{\pi}(\mathbf{s}_{t},\mathbf{a}_{t})^{\top}\mathbf{w}$.

Subsequently, the policy $\pi'$ may be recovered
$$
\pi'(\mathbf{s}) = \underset{\mathbf{a}}{\arg\max}\ \psi^{\pi}(\mathbf{s},\mathbf{a})^{\top}\mathbf{w}
$$
which represents *not* the optimal policy, but the policy recovered after one step of policy iteration! (Since the $Q$-function represents an evaluation of the *current* policy $\pi$).

**Idea 2:** recover many $Q$-functions.

1. Train $\psi^{\pi_{k}}(\mathbf{s}_{t},\mathbf{a}_{t})$ for many policies $\pi_{k}$.
2. Get some reward samples $\{ \mathbf{s}_{i},r_{i} \}$.
3. Optimize $\mathbf{w}\leftarrow \arg\min_{\mathbf{w}}\sum_{i}\lVert \phi(\mathbf{s}_{i})^{\top}\mathbf{w}-r_{i} \rVert^{2}$.
4. Recover $Q^{\pi_{k}}(\mathbf{s}_{t},\mathbf{a}_{t})\approx \psi^{\pi_{k}}(\mathbf{s}_{t},\mathbf{a}_{t})^{\top}\mathbf{w}$ for every $\pi_{k}$.

Then, we recover the policy $\pi'$ as
$$
\pi'(\mathbf{s})=\arg\max _{\mathbf{a}}\max _{k} \psi^{\pi_{k}}(\mathbf{s},\mathbf{a})^{\top}\mathbf{w}
$$
and this is now much better than the policy produced from idea 1. It won't produce the optimal policy, but it will find the highest reward policy in each state, resulting in more substantial improvements.
### Continuous Successor Representations
The issue with successor representations in a continuous state space is that, in the equation
$$
\begin{align*}
\mu_{i}^{\pi}(\mathbf{s}_{t}) = (1-\gamma)\delta(\mathbf{s}_{t}=i)+\gamma \mathbb{E}_{\mathbf{a}_{t}\sim \pi(\mathbf{a}_{t}\mid \mathbf{s}_{t}),\mathbf{s}_{t+1}\sim p(\mathbf{s}_{t+1}\mid \mathbf{s}_{t},\mathbf{a}_{t})}[\mu_{i}^{\pi}(\mathbf{s}_{t+1})]
\end{align*}
$$
$\delta(\mathbf{s}_{t}=i)$ is zero for any sampled state. Thus, we must reframe successor representation as a **classification** problem. We define a binary classifier $p^{\pi}(F=1\mid \mathbf{s}_{t},\mathbf{a}_{t},\mathbf{s}_{\text{future}})$ where $F=1$ indicates that $\mathbf{s}_{\text{future}}$ is a future state from $\mathbf{s}_{t},\mathbf{a}_{t}$ under $\pi$. We can easily construct a training set for this classifier with $\mathcal{D}_{+}$, the positive examples, sampled from all states reachable from $\mathbf{s}_{t},\mathbf{a}_{t}$,  i.e. $D_{+}\sim p^{\pi}(\mathbf{s}_{\text{future}}\mid \mathbf{s}_{t},\mathbf{a}_{t})$, and $D_{-}$, the negative examples, sampled from all states reachable by the policy, i.e. $D_{-}\sim p^{\pi}(\mathbf{s})$. (Note that the probability that a state in $D_{-}$ is in $D_{+}$ is negligible since the state space is continuous). We define the *Bayes' optimal classifier* by
$$
p^{\pi}(F=1\mid \mathbf{s}_{t},\mathbf{a}_{t},\mathbf{s}_{\text{future}}) = \frac{p^{\pi}(\mathbf{s}_{\text{future}}\mid \mathbf{s}_{t},\mathbf{a}_{t})}{p^{\pi}(\mathbf{s}_{\text{future}}\mid \mathbf{s}_{t},\mathbf{a}_{t})+p^{\pi}(\mathbf{s}_{\text{future}})} \\
$$
from which we can derive an expression for $p^{\pi}(\mathbf{s}_{\text{future}}\mid \mathbf{s}_{t},\mathbf{a}_{t})$, which is precisely the **successor representation**!
$$
\begin{align*}
p^{\pi}(F=1\mid \mathbf{s}_{t},\mathbf{a}_{t},\mathbf{s}_{\text{future}}) &= \frac{p^{\pi}(\mathbf{s}_{\text{future}}\mid \mathbf{s}_{t},\mathbf{a}_{t})}{p^{\pi}(\mathbf{s}_{\text{future}}\mid \mathbf{s}_{t},\mathbf{a}_{t})+p^{\pi}(\mathbf{s}_{\text{future}})} \\
p^{\pi}(F=0\mid \mathbf{s}_{t},\mathbf{a}_{t},\mathbf{s}_{\text{future}}) &= \frac{p^{\pi}(\mathbf{s}_{\text{future}})}{p^{\pi}(\mathbf{s}_{\text{future}}\mid \mathbf{s}_{t},\mathbf{a}_{t})+p^{\pi}(\mathbf{s}_{\text{future}})} \\
\frac{p^{\pi}(F=1\mid \mathbf{s}_{t},\mathbf{a}_{t},\mathbf{s}_{\text{future}})}{p^{\pi}(F=0\mid \mathbf{s}_{t},\mathbf{a}_{t},\mathbf{s}_{\text{future}})} p^{\pi}(\mathbf{s}_{\text{future}}) &= p^{\pi}(\mathbf{s}_{\text{future}}\mid \mathbf{s}_{t},\mathbf{a}_{t})
\end{align*}
$$
Note that $p^{\pi}(\mathbf{s}_{\text{future}})$ is a bit mysterious and difficult to determine, but it is a *constant* with regards to $\mathbf{s}_{t},\mathbf{a}_{t}$ and therefore unimportant for learning the $Q$-function.

From this, we may derive the **C-Learning** algorithm, from Eysenbach et al.'s paper *C-Learning: Learning to Achieve Goals via Recursive Classification*.

1. Sample $\mathbf{s}\sim p^{\pi}(\mathbf{s})$ (e.g. run policy, sample trajectories)
2. Sample $\mathbf{s}\sim p^{\pi}(\mathbf{s}_{\text{future}}\mid \mathbf{s}_{t},\mathbf{a}_{t})$ (e.g. pick $\mathbf{s}_{t'}$ where $t'=t+\Delta,\Delta \sim\text{Geom}(\gamma)$)
3. Update $p^{\pi}(F=1\mid \mathbf{s}_{t},\mathbf{a}_{t},\mathbf{s})$ using SGD with cross-entropy loss.

Note that the above is an **on-policy** algorithm.
## Hierarchical RL
The classical formulation of hierarchical RL revolves around **options**. Options are like a mini-policy or skill, and are described by
$$
\mathbf{o}=(I_{\mathbf{o}},\pi_{\mathbf{o}},\beta_{\mathbf{o}})
$$
where $\pi_{\mathbf{o}}$ is the option policy (effectively just a regular RL policy), $I_{\mathbf{o}}$ is the initiation set, the set of the states where the option policy may begin, and $\beta_{\mathbf{o}}$ is the termination set, defined analogously.

Then, we extend the action space with an *option space* $\mathcal{O}$, i.e. we define $\mathcal{A}':=\mathcal{O}\cup \mathcal{A}$, and have the policy learn $\pi_{\theta}(\mathbf{o}\mid \mathbf{s})$ and $Q(\mathbf{s},\mathbf{o})$. Now, our $Q$-function backup looks like
$$
Q(\mathbf{s}_{t},\mathbf{o}_{t}) \leftarrow \left[\sum_{t=t'}^{t+h-1} r(\mathbf{s}_{t'})\right]+\gamma^{h}\max _{\mathbf{o}_{t+h}:\mathbf{s}_{t+h}\in I_{\mathbf{o}_{t}+h}}Q(\mathbf{s}_{t+h},\mathbf{o}_{t+h})
$$
This is potentially very good if $h$ is large, as it effectively shortens the entire horizon $H$ of the task.

But, how do we learn these options? After all, learning options end-to-end alongside the policy would likely negate much of the benefit. Well, if the options themselves have some special structure that is easier to learn then the overall, high-level policy, then learning these options can prove very useful. This, then, leads us to the modern, more practical frameworks for hierarchical RL.

One approach is hierarchical RL via multi-task policies. Rather than treating each option as a separate policy, we group the options together and train one multi-task policy conditioned on the option, i.e. $\pi_{\mathbf{o}}=\pi(\mathbf{a}\mid \mathbf{s},\mathbf{o})$. We additionally disregard the notion of initiation and termination states, and simply run each option policy for some constant number of steps. From this framework, we can create the following simple algorithm.

1. Train $\pi_{\text{lo}}(\mathbf{a}\mid \mathbf{s},\mathbf{o})$ using skill discovery.
2. Train $\pi_{\text{hi}}(\mathbf{o}\mid \mathbf{s})$ with regular RL, switching to a different $\mathbf{o}$ every $k$ steps.

Our $Q$-function backup then looks like
$$
Q(\mathbf{s}_{t},\mathbf{o}_{t}) \leftarrow \left[\sum_{t=t'}^{t+h-1} r(\mathbf{s}_{t'})\right]+\gamma^{k}\max _{\mathbf{o}_{t+h}}Q(\mathbf{s}_{t+k},\mathbf{o}_{t+h})
$$
<div style="page-break-after: always;"></div>

