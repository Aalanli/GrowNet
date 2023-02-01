- [ ] # GrowNet
## Timeline
While this project originally was created to test my ideas in growing neural net architectures, involving sparsity, it has since grown (intended :), mainly fueled by the richness in the design space, into a simulation suite of sorts. 
A lot of what is currently in place is the scafolding for a simulation gui built on top of the bevy game engine, in rust. As current ml-tools for visualization in python are too slow, and julia makie is not flexible enough for gui development. 
The core idea is around the concept of associating the nodes in a computation graph to a physical space, akin to the neurons of the brain embedded in $\mathbb{R}^3$, so that this extra structure gives better properties in inducing sparsity and causality of information flow. As well, this setup is well attuned to visualization, as individual neurons are low-dimensional. But since then, lots of new architectures has come to my attention, and I envision this project capturing them and expanding them.

There are three core problems this project aims to tackle
1. How to decompose losses into subtasks, to tackle the credit assignment problem
2. In a related note, how to propagate errors through the network without back-propagation
3. Related, how to model long temporal dynamics without storing intermediate activations, perhaps in a biological plausible way, or at least amendable to efficient analogue hardware

Additional problems
1. What is the intersection between discrete symbols (in symbolic AI) and continuous representations in deep learning
2. How to make distributed representations (continous) more interpretable
3. Continual learning, transfer learning and tackling catastrophic forgetting
4. Internal world model

# Ideas
To tackle these three problems, I picture the potential solution as decomposed into two categories, that of the structured approach and the unstructured approach.

The structured approach would involve some explict formulation of a backwards and forwards pass, of the forward computation and backwards adjustment of weights, perhaps as explict as back-propagation in the supervised setting. Generally this would involve two graphs $\hat{y} = f(\theta, x)$ and $\Delta \theta = g(\gamma,\theta, y, \hat{y})$, with $f$ the forwards graph, and $g$ the backwards pass.
This basically encompasses all current approaches in deep learning and some other more exotic methods.

The unstructured category currently is far more nebulous, the best example I can think of is [reservoir computing](https://julien-vitay.net/lecturenotes-neurocomputing/4-neurocomputing/4-Reservoir.html), where the resevoir is a non-parametric function, essentially a glorified activation function which specifies some complex dynamics. The inputs are fed into this blackbox reservoir and outputs are read and fed into a standard dl model, trained via back-prop. This approach is unpleasant to me for several reasons, namely that the representation power of the entire model is based on top of the non-linear reservoir, of which is sensitive to hyperparameters since it is not trained. Ideally, the unstructured category would entail training this reservoir via a local learning rule, from which back-prop or what ever is happening in the brain emerges, without the explicit formulation of a *dual graph* as seen in the structured approach.
Some examples of this are [hebbian learning](https://julien-vitay.net/lecturenotes-neurocomputing/4-neurocomputing/5-Hebbian.html), [STDP](https://en.wikipedia.org/wiki/Spike-timing-dependent_plasticity).

## 1. Modeling long sequences
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

## 2. Credit Assignment
To tackle the credit assignment problem, I imagined 
[Decoupled Neural Interfaces using Synthetic Gradients](https://arxiv.org/pdf/1608.05343.pdf)
Basically intermediary connections/shortcuts in the backwards gradient graph, and optionally forward shortcutting. Basically has an auxillary network which predicts the gradient, and gets trained periodically based off the actual gradient.

This line of research is quite interesting, as it shows that the 'gradient', or the direction to minimize the loss, is not strictly equal to the derivative of the function w.r.t. the loss. 
Even random matrices will work. If the forward pass is $y = Wx$, then the backwards is $\frac{\partial \mathcal{L}}{\partial x} = W^T \frac{\partial \mathcal{L}}{\partial y}$, and $\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial y} x^T$. But a random matrix $W'$ could also work for $\frac{\partial \mathcal{L}}{\partial x}$.

### Decomposing Subtasks
*Cavet: This section consists of my own thoughts, however flawed. I have cited all works which I have read, and do not claim these thoughts to be original or rigorous; this is just for fun :)*

We say that a task $T$ consists of a behavior $B: X \to \hat Y$ and an objective $\mathcal{L}: \hat Y \times Y \to \mathbb{R}$ and samples of the task $S = \{(x, y) : x \in X, y \in Y\}$. Then to satisfy the task, we must find $$B^* = \text{argmin}_B \sum_i \mathcal{L}(B(x_i), y_i)$$
Or *(whatever the syntax is...)*$$B^* = \text{argmin}_B \mathbb{E}_{(x, y)}[\mathcal{L}(B(x), y)]$$
So naturally, $x \xrightarrow{T} \hat y$ is the notation for a certain task. Obviously, all of what I said is just supervised machine learning. One may wonder if it would be possible to decompose a task into subtasks such that each the network of tasks form a tree, with supertasks and subtasks. Or perhaps more generally, a directed graph, such that gradients do not pass between task boundaries and information between tasks can only be passed locally. For example, in image classification, one finds that the layers naturally separate into tasks, with the task of the lower layers to detect edges, middle layers detecting shapes/textures, and upper layers detecting ... (I don't know what upper layers detect). On that topic, perhaps this formulation can help with interpretability, with the following definition? 

Each $\mathcal{L}$ can be decomposed into $\mathcal{R}: Y \to \tilde Y$ and $\mathcal{D}: \hat Y \times \tilde Y \to \mathbb{R}$, such that $\mathcal{L}(\hat y, y) = \mathcal{D}(\hat y, \mathcal{R}(y))$. Then we call $\mathcal{R}$ the representative of task $T$, which gives the expected output $\tilde y$ of behavior $B$, from the required input $y$ of an upper task $T^\prime$. Then $\mathcal{D}$ is just some distance function, like $||\cdot||_2$. 

Then, say task $T$ is too hard; the behavior $B$ cannot some constraints of task $T$. Then we ask is it possible to split task $T$ into $T_1$ and $T_2$, such that the composition of the two behaviors satisfy the original task? 
$$x \xrightarrow{T} y \equiv x \xrightarrow{T_1} \tilde y_1 \xrightarrow{T_2} y$$
Then how to we pick $\tilde y_1$? How to optimize this thing?

![Diagram1](https://github.com/Aalanli/GrowNet/blob/main/Diagram1.png)

[link](https://tikzcd.yichuanshen.de/#N4Igdg9gJgpgziAXAbVABwnAlgFyxMJZABgBpiBdUkANwEMAbAVxiRAA8QBfU9TXfIRQBGclVqMWbADrSAFnRwACAJ4B9Yd14gM2PASIAmMdXrNWiELLwNYqjVr57BRAMwmJ5mfMX3DjnX59IWQAFg8zKUsVAN0BAxRjYXFIiytpAFtFOQBjRmAAJS41fy5xGCgAc3giUAAzACcIDKQyEBwIJFFPKPSsnFz8gCENLhBqBjoAIxgGAAUgl0sGGDqcAMbmruoOpGMetNl+wYZgABFizQnp2YXnBJAVtfGQORg6KDZIMFZqGbBPohXMQePUmi1EPtdkDTJJDplsnlTiNDGNrjN5osHk91qCQJsIe52p1EABWWFeRBgJgMBjo25YoSPVa47QEpBE6HhA7eY5I87Ffz0zH3Jk4l5vD5fAi-ED-QHAvHssk7En7VJIam04V3eJilncChcIA)

*Kind of awkward, but Obsidian does not have commutative diagrams*
Note that $\mathcal{R}_2 = \text{id}$, and is omitted, since task2's representative is just $y$.

So we train $\mathcal{B}_1$ with standard gradient descent on $\mathcal{D_1}$ letting $\tilde y_1$ be the 'label', if it were supervised learning.
$$\hat y_1 = \mathcal{B}_1(\theta_1, x)$$ 
$$\tilde y_1 = \mathcal{R}_1(\theta_{r_1},y)$$ 
$$\hat y_2 = \mathcal{B}_2 (\theta_2, \tilde y_1)$$

$$L_1 = \mathcal{D}_1(\hat y_1, \tilde y_1) = ||\hat y - \tilde y_1 ||_2$$

$$L_2 = \mathcal{D}_2(\hat y_2, y) = ||\hat y_2 - y ||_2$$


Then,
$$\frac{\partial L_1}{\partial \theta_1} = \frac{\partial L_1}{\partial \hat y_1}\frac{\partial\hat y_1}{\partial \theta_1}$$
$$\frac{\partial L_2}{\partial\theta_2} = \frac{\partial L_2}{\partial \hat y_2}\frac{\partial\hat y_2}{\partial \theta_2}$$

and $$L_{r_1} = d(L_1, L_2)$$
Where $d$ is some function which ideally minimizes both $L_1$ and $L_2$.

$$\frac{\partial L_1}{\partial \theta_{r_1}} = \frac{\partial L_{r_1}}{\partial L_1}\frac{\partial L_1}{\partial \tilde y_1}\frac{\partial \tilde y_1}{\partial \theta_{r_1}} + \frac{\partial L_{r_1}}{\partial L_2}\frac{\partial L_2}{\partial \tilde y_1}\frac{\partial \tilde y_1}{\partial \theta_{r_1}}$$

So that the the maximum depth of gradient computation is $\mathcal{B_2}$, which already is happening anyways, we just need $\frac{\partial L_2}{\partial \tilde y_1}$

## Relevant sources
[Towards Biologically Plausible Deep Learning](https://arxiv.org/pdf/1502.04156.pdf)
[Decoupled Neural Interfaces using Synthetic Gradients](https://arxiv.org/pdf/1608.05343.pdf)
[Beyond Backprop: Online Alternating Minimization with Auxiliary Variables](https://arxiv.org/pdf/1806.09077.pdf)
[Difference Target Propagation](https://arxiv.org/pdf/1412.7525.pdf)
[Deep Learning With Spiking Neurons: Opportunities and Challenges](https://www.frontiersin.org/articles/10.3389/fnins.2018.00774/full)
[The Forward Forward Algorithm](https://arxiv.org/abs/2212.13345)
[Biologically inspired alternatives to backpropagation through time](https://arxiv.org/abs/1901.09049)



