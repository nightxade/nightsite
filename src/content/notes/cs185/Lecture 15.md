# Lecture 15: Model-Based RL
## Learning a Simulator
Model-based RL, in contrast to the model-free RL methods discussed thus far, attempts to learn a model of the problem environment in order to produce an optimal policy. Here's a sketch of a simple simulator-learning algorithm.

1. Sample data $\mathcal{D}=\{ \mathbf{s}_{i},\mathbf{a}_{i},\mathbf{s}_{i}' \}$ using base policy $\pi_{\beta}(\mathbf{a}\mid \mathbf{s})$.
2. Learn simulator $f(\mathbf{s},\mathbf{a})=\mathbf{s}'$ with some loss function, e.g. MSE or MLE.
3. Run favorite RL method inside simulator $f$. This could simply be a model-free algorithm, or it could be a planning/trajectory optimization algorithm that takes advantage of our simulator $f$.

There can be some complications, however, with each step.

1. The dataset must be somewhat comprehensive with regards to the state space, thus the choice of $\pi_{\beta}$ is extremely important to the consequent dataset quality. (Statistics/algorithms problem).
2. The design of the simulator model is *domain-dependent* since the model complexity and structure depend heavily on the problem complexity and structure. (Deep learning/models problem).
3. In general, model-free RL in the learned simulator may be difficult; for instance, the simulator may be very computationally inefficient, making sampling expensive. (Controls/RL problem).

Today, we'll primarily discuss the first step/problem.
## Distributional Shift and Uncertainty
Let $\pi_{f}$ be the optimal policy learned under the simulator $f$. The key issue with problem 1 is that the state distribution observed in the dataset produced by $\pi_{\beta}$ is not the same as the state distribution observed by $\pi_{f}$; in other words, we have a **distributional shift** problem, just like in [[college/sp26/classes/cs185/notes/Lecture 2|Lecture 2]] with imitation learning!

Well, what if we add the following step to our algorithm?

1. Sample data $\mathcal{D}=\{ \mathbf{s}_{i},\mathbf{a}_{i},\mathbf{s}_{i}' \}$ using base policy $\pi_{\beta}(\mathbf{a}\mid \mathbf{s})$.
2. Learn simulator $f(\mathbf{s},\mathbf{a})=\mathbf{s}'$. 
3. Learn optimal policy $\pi_{f}$ under $f$.
4. $\pi_{\beta}\leftarrow \pi_{f}$.

> [!tip]
> Note the similarity between this modification and [[college/sp26/classes/cs185/notes/Lecture 2|DAgger]].

Does this fix the problem? Unfortunately, no; while this is an improvement, this is not sufficient. In particular, step 3, which learns the optimal policy $\pi_{f}$, will actually end up *exploiting the mistakes* or misrepresentations in our simulator $f$. This encourages extreme policies, which massively slows down training speed not only because it produces policies with bad generalization but also because those policies will produce wildly varying data.

So, what if we replace step 3 with just a small improvement to our policy, like in [[Lecture 9#Proximal Policy Optimization|PPO]]? This is a good idea, but what does "small" mean here? Ideally, we'd like larger improvements to learn faster, while also being sufficiently small such that our model does not overfit to our erroneous simulator $f$. Generally speaking, answering this question is a matter of hyperparameter tuning during training.

Also, how exactly do we enforce this "small" improvement? We have a few options.

- Add a trust region ([[Lecture 10#Trust Region Policy Optimization|TRPO]]) or KL divergence constraint $D_{\text{KL}}(\pi_{f}\parallel \pi_{\beta})\leq\epsilon$.
- Stick to regions where the simulator model is "confident." This may be implemented with a *probabilistic*, uncertainty-aware model or a pessimism penalty when presented with any model uncertainty. For the first, in step 3, we additionally extract the best policy *in expectation* under our probabilistic model.

For the probabilistic model, there are notably a few caveats to consider.

- We'd like to promote exploration when collecting our dataset to better improve our simulator model.
- Extracting the best policy in expectation is not the same as a pessimistic/optimistic value.
## Uncertainty-Aware Neural Networks
Consider first a neural network that outputs a distribution over the state space. What if we use the *entropy* of the output distribution to measure confidence? Unfortunately, this is problematic because the simulator model often becomes erroneously confident due to excessive overfitting.

In particular, there are really two types of uncertainty.

- **Aleatoric** or statistical uncertainty: "how random does the model think the data is?"
- **Epistemic** or model uncertainty: "the model is certain about the data, but we are uncertain about the model itself!"

Epistemic uncertainty is what we are trying to reduce; let's try and estimate it!

In typical MLE, we estimate $\hat{\theta}=\arg\max_{\theta}\log p(\theta \mid \mathcal{D})$. However, what if, instead of learning just a single, likelihood-maximizing parameter $\hat{\theta}$, we learned the *conditional distribution* of the parameters, i.e. we learned $p(\theta \mid \mathcal{D})$. This distribution's entropy captures our epistemic uncertainty—if we're certain about the model, the distribution will have extremely low entropy, with most probability mass being concentrated on a single point. In contrast, if we're uncertain, the distribution will have high entropy.

Moreover, with $p(\theta \mid \mathcal{D})$, we can make predictions based on the entire distribution of our parameters, i.e.
$$
\int p(\mathbf{s}_{t+1}\mid \mathbf{s}_{t},\mathbf{a}_{t})p(\theta \mid \mathcal{D}) \,\mathrm{d}\theta 
$$
Learning an entire distribution is generally intractable/much harder than learning a single parameter estimate—in practice, we approximate the distribution.

One such approximation a **Bayesian neural network**. A Bayesian neural network replaces weight parameters with distributions over each weight parameter. This corresponds to a **fully factorized posterior**, as we are approximating our distribution $p(\theta \mid \mathcal{D})$ with $\prod_{i}p(\theta_{i}\mid \mathcal{D})$, which includes a posterior distribution for every individual parameter. Notably, this does not allow representing covariances between parameters; but it is simple. Also, we typically represent $p(\theta_{i}\mid \mathcal{D})=\mathcal{N}(\mu_{i},\sigma_{i}^{2})$, and thus the number of parameters really only doubles. And, how are they trained? Variational inference. We won't discuss the derivation of this in detail.

Another, simpler approximation is **bootstrap ensembles**, in which we learn multiple models, and synthesize them to produce a prediction $p(\theta \mid \mathcal{D})\approx \frac{1}{N}\sum_{i}\delta(\theta_{i})$. (See [[7. Regularization for Deep Learning#7.11 Bagging and Other Ensemble Methods|chapter 7 from Goodfellow et al.]]). But how do we train these models so that they don't simply learn the same parameters? We could partition our dataset into $k$ disjoint sets for $k$ models; however, this may be a very inefficient use of our dataset. Instead, we can choose to produce $\mathcal{D}_{i}$ by sampling with replacement from $\mathcal{D}$ for the same size as the number of data points in $\mathcal{D}$. The "with replacement" sampling produces sufficiently independent datasets that translate to sufficiently independent models.

In practice, bootstrap ensembles operate with a couple caveats/considerations: $(1)$ the number of models is typically small for efficiency reasons, which does lower approximation accuracy, and $(2)$ SGD with random parameter initialization typically produces sufficiently independent models without resampling with replacement.
<div style="page-break-after: always;"></div>

