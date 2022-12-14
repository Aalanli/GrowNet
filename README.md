# GrowNet
## Idea/timeline
While this project originally was created to test my ideas in growing neural net architectures, involving sparsity, it has since grown (intended :), mainly fueled by the richness in the design space, into a simulation suite of sorts. 
A lot of what is currently in place is the scafolding for a simulation gui built on top of the bevy game engine, in rust. As current ml-tools for visualization in python are too slow, and julia makie is not flexible enough for gui development. 
The core idea is around the concept of associating the nodes in a computation graph to a physical space, akin to the neurons of the brain embedded in $\mathbb{R}^3$, so that this extra structure gives better properties in inducing sparsity and causality of information flow. As well, this setup is well attuned to visualization, as individual neurons are low-dimensional. But since then, lots of new architectures has come to my attention, and I envision this project capturing them and expanding them.

There are three core problems this project aims to tackle
1. How to decompose losses into subtasks, to tackle the credit assignment problem
2. In a related note, how to propagate errors through the network without back-propagation
3. Related, how to model long temporal dynamics without storing intermediate activations, perhaps in a biological plausible way, or at least amendable to efficient analogue hardware

## Ideas
To tackle these three problems, I picture the potential solution as decomposed into two categories, that of the structured approach and the unstructured approach.

The structured approach would involve some explict formulation of a backwards and forwards pass, of the forward computation and backwards adjustment of weights, perhaps as explict as back-propagation in the supervised setting. Generally this would involve two graphs $\hat{y} = f(\theta, x)$ and $\Delta \theta = g(\gamma,\theta, y, \hat{y})$, with $f$ the forwards graph, and $g$ the backwards pass.
This basically encompasses all current approaches in deep learning and some other more exotic methods.

The unstructured category currently is far more nebulous, the best example I can think of is [reservoir computing](https://julien-vitay.net/lecturenotes-neurocomputing/4-neurocomputing/4-Reservoir.html), where the resevoir is a non-parametric function, essentially a glorified activation function which specifies some complex dynamics. The inputs are fed into this blackbox reservoir and outputs are read and fed into a standard dl model, trained via back-prop. This approach is unpleasant to me for several reasons, namely that the representation power of the entire model is based on top of the non-linear reservoir, of which is sensitive to hyperparameters since it is not trained. Ideally, the unstructured category would entail training this reservoir via a local learning rule, from which back-prop or what ever is happening in the brain emerges, without the explicit formulation of a *dual graph* as seen in the structured approach.
Some examples of this are [hebbian learning](https://julien-vitay.net/lecturenotes-neurocomputing/4-neurocomputing/5-Hebbian.html), [STDP](https://en.wikipedia.org/wiki/Spike-timing-dependent_plasticity).

### 1. Modeling long sequences
An initial idea in modeling long sequence with low memory budgets involves foraying into bijective model architectures. If the model has an inverse available, then storing intermediate activations becomes unnecessary as one could simply unroll from the output.
For example, in a simplified recurrent neural network $$\begin{align} h_t &= f(W_h, h_{t-1}, x_t) \\ \hat y_t &= g(W_y, h_t) \end{align}$$
where $h_t$ is the hidden state, $W_h$ are the parameters for computing the hidden state and $W_y$ for the output. And the loss at time $t$ is $L_t = \mathcal{L}(\hat y_t, y)$. Suppose the network computes outputs from $t$ to $T$.
Then $$\frac{\partial L_T}{\partial W_h} = \frac{\partial L_T}{\partial \hat y_T} \frac{\partial \hat y_T}{\partial h_T} \frac{\partial h_T}{\partial h_{T-1}} \dots \frac{\partial h_{t+1}}{\partial h_{t}}$$
With each $\frac{\partial h_{i}}{\partial h_{i-1}}$ requiring the storage of intermediate $h_i$. But if the inverse $h_{t-1} = f^{-1}(W_h, h_t, x_t)$ exists, then the previous activation can be unrolled and computed in reverse, incurring $\mathcal{O}(1)$ memory cost and $\mathcal{O}(n)$ compute cost, compared to the $\mathcal{O}(n^2)$ compute cost of otherwise computing $h_t, h_{t+1}, \dots h_{t+n}$ from the beginning.

This is very similar to [neural odes](https://arxiv.org/abs/1806.07366), which employs the adjoint trick to unroll in reverse from the forward $\frac{d h(t)}{dt} = f(\theta, h(t), x(t), t)$ ode. Except its more numerically stable ([ANode](https://arxiv.org/pdf/1902.10298.pdf) suggested that adjoint trick is not suitable for stiff odes)?

Obviously constraining networks to be invertiable is non-trivial, as is often the case, the inverse is more numerically unstable and computational intensive, additionally, many problems in ml, such as image classification is not bijective.

One can think of constructing an approximate inverse $\hat y = f(\theta_1, x), g(\theta_2, \hat y) \approx f^{-1}$ s.t. $||g(\theta_2, f(\theta_1, x)) - x||^2_2 \leq \epsilon$?
Then $g$ can be trained using supervised gradient descent. This is closely linked to the explicit formulation of the categorical method, with a forward graph and its dual.

Another approach lies in unitary matrices, that of constraining transformations to be purely unitary, ([EUNN](https://arxiv.org/abs/1612.05231)). Or further constrain the space to a FFT like transform, with efficient implementations in hardware. This critically allows efficient inverses, as well as further efficiencies in computing large powers, as is normally the case in the backwards pass.

For amending the bijective nature of these networks to notably not bijective tasks like classification, one can think of 'dummy dimensions', where we only train and care about a subset of the output dimensions, and interpret those to be the labels. Otherwise, the network is purely isotropic.

