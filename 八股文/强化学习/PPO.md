# PPO算法

PPO（Proximal Policy Optimization，近端策略优化）算法是一种在强化学习领域广泛应用的策略优化算法。它在2017年由John Schulman等人提出，是TRPO（Trust Region Policy Optimization，信任域策略优化）算法的改进版本，旨在解决TRPO计算复杂度高、实现困难的问题，同时保持良好的性能。PPO算法在许多实际应用中表现出色，尤其是在机器人控制、游戏AI等领域。

## 1. **PPO算法的核心思想**

PPO算法是一种基于策略梯度的方法，其目标是通过优化策略函数 $ \pi(a|s) $ 来最大化累积奖励。策略函数表示在状态 $ s $ 下选择动作 $ a $ 的概率分布。PPO的核心思想是通过更新策略函数，使得新的策略在一定程度上接近旧的策略，从而保证更新的稳定性。

PPO算法的关键在于引入了**截断的概率比**（clipped probability ratio）和**信任域约束**（trust region constraint）。具体来说，PPO通过限制策略更新的幅度来避免过度更新，从而保证更新过程的稳定性和收敛性。

**解决了什么问题**：

- 1.**策略更新不稳定**：

传统策略梯度方法（如REINFORCE）在优化过程中可能会导致**策略更新过大，策略更新不稳定**，从而引发训练过程中的剧烈波动，甚至导致策略崩溃。PPO通过引入剪切（Clipping）机制，限制新旧策略之间的差异，确保策略更新在一个可控范围内，从而**提高训练的稳定性**

- 2.**样本效率低**：

传统方法中，每个数据样本通常只用于一次梯度更新，导致样本利用率较低。PPO通过重要性采样（Importance Sampling）技术，允许重复利用历史数据，从而提高样本效率。

## 2. **PPO算法的理论的前置知识**

好的！在介绍PPO算法之前，我们先来解释一下强化学习中的一些基本概念。这些概念是理解PPO算法的基础。

### 1. **环境（Environment）**

环境是智能体所处的外部世界。环境根据智能体的动作给出反馈，包括新的状态（State）和奖励（Reward）。

### 2. **状态（State）**

状态是环境在某一时刻的描述。智能体根据当前状态选择动作。状态可以是离散的（如网格世界中的位置）或连续的（如机器人的关节角度）。

### 3. **动作（Action）**

动作是智能体在某一时刻根据状态做出的选择。动作可以是离散的（如“向上”“向下”）或连续的（如机器人的关节速度）。

### 4. **策略网络（Policy）**

策略是智能体的行为规则，它定义了在给定状态下选择动作的概率分布。策略通常用 $\pi(a|s,\theta)$ 表示，即在状态 $s$ 下选择动作 $a$ 的概率。我们可以使用深度学习的神经网络来表示策略网络。

### 5. **奖励（Reward）**

奖励是环境对智能体动作的反馈，是一个标量值。奖励信号告诉智能体其动作的好坏。智能体的目标是最大化累积奖励。

### 6. **累积奖励（Cumulative Reward）**

累积奖励是指智能体在一系列动作中获得的总奖励。通常用折扣累积奖励（Discounted Cumulative Reward）来表示，公式为：
$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots
$$
其中，$\gamma$ 是折扣因子（$0 \leq \gamma \leq 1$），用于平衡短期和长期奖励。

### 7. **价值函数（Value Function）**

价值函数评估在某个策略下，某个状态或状态-动作对的“价值”。

- **状态价值函数（State Value Function）**：$V_\pi(s)$ 表示在策略 $\pi$ 下，从状态 $s$ 开始的期望累积奖励。

$$
V_\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]
$$

- **动作价值函数（Action Value Function）**：$Q_\pi(s, a)$ 表示在策略 $\pi$ 下，从状态 $s$ 开始并选择动作 $a$ 的期望累积奖励。

$$
Q_\pi(s, a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]
$$

### 8. **优势函数（Advantage Function）**

优势函数 $A(s, a)$ 表示在状态 $s$ 下选择动作 $a$ 的优势，即：
$$
A(s, a) = Q(s, a) - V(s)
$$
它衡量了选择动作 $a$ 相对于平均动作的价值。

### 9. **策略梯度（Policy Gradient）**

策略梯度是强化学习中一种优化策略的方法。通过计算策略的梯度来更新策略参数，使得期望累积奖励最大化。策略梯度的基本公式为：
$$
\nabla_\theta J(\theta) = \mathbb{E}_\pi[\nabla_\theta \log \pi(a|s) \cdot G_t]
$$
其中，$J(\theta)$ 是目标函数，$\theta$ 是策略参数。

### 10. **演员-评论家方法（Actor-Critic Method）**

演员-评论家方法是一种强化学习算法框架，包含两个部分：

- **演员（Actor）**：负责根据当前策略选择动作。
- **评论家（Critic）**：负责评估当前策略的价值函数或优势函数，为演员提供更新信号。

### 11. **KL散度（KL Divergence）**

KL散度是衡量两个概率分布之间差异的指标。PPO算法通过限制新策略与旧策略之间的KL散度，确保策略更新的稳定性。

## 3. 策略梯度定理

我们详细推导梯度策略算法中损失函数的表达式。这个推导过程将从策略梯度的基本定理开始，逐步引入采样和平均，最终得到损失函数的表达式。

### 1. 策略梯度定理

策略梯度定理提供了一种计算策略参数梯度的方法，该梯度指向期望回报增加的方向。对于参数化策略 $\pi_\theta(a|s)$，其梯度可以表示为：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot R_t \right]
$$

其中：

- $J(\theta)$ 是策略的期望回报。
- $\tau = (s_0, a_0, s_1, a_1, \ldots, s_T)$ 是一个完整的轨迹，包括状态、动作和时间步。
- $R_t$ 是在时间步 $t$ 获得的奖励。
- $\pi_\theta(a_t | s_t)$ 是在状态 $s_t$ 下采取动作 $a_t$ 的概率。

### 2. 期望回报的最大化

为了最大化期望回报 $J(\theta)$，我们需要沿着梯度的方向更新策略参数 $\theta$。这可以通过梯度上升实现：

$$
\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)
$$

其中 $\alpha$ 是学习率。

### 3. 采样和平均

在实际应用中，直接计算期望值通常是不可行的，因为我们无法遍历所有可能的轨迹。因此，我们通过采样 $N$ 个独立的轨迹来估计期望值。每个轨迹的长度可能不同，设为 $T_n$。

对于每个采样的轨迹 $\tau^n$，我们可以计算其总奖励 $R(\tau^n)$，然后使用这些轨迹来估计梯度：

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{n=1}^N \sum_{t=0}^{T_n} \nabla_\theta \log \pi_\theta(a_t^n | s_t^n) \cdot R(\tau^n)
$$

这里，$a_t^n$ 和 $s_t^n$ 分别是第 $n$ 个轨迹在时间步 $t$ 的动作和状态。

### 4. 构建损失函数

为了优化策略参数 $\theta$，我们需要最小化损失函数。损失函数通常被定义为策略梯度的负期望值：

$$
L(\theta) = -\mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \log \pi_\theta(a_t | s_t) \cdot R_t \right]
$$

通过采样和平均，我们可以估计这个期望值：

$$
L(\theta) \approx -\frac{1}{N} \sum_{n=1}^N \sum_{t=0}^{T_n} \log \pi_\theta(a_t^n | s_t^n) \cdot R(\tau^n)
$$

### 5. 损失函数的最终形式

最终，我们得到损失函数的表达式：

$$
\text{Loss} = -\frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n} R(\tau^n) \log \pi_\theta(a_n^t | s_n^t)
$$

这个损失函数用于梯度下降或上升算法中，以更新策略参数 $\theta$，从而提高策略的性能。

### 5.策略梯度来训练策略网络

以马里奥游戏为例，我们首先将游戏的画面当作环境输入到策略网络中，策略网络会输出动作的概率，我们从中采样并且获得动作、执行动作得到新的环境。我们不断重复这个流程，我们就可以得到N个动作和奖励，可以作为一个batch来进行参数更新。我们采集数据和更新数据用的都是同一个模型，我们称之为**On Policy**。他有一个问题，我们的大部分时间都在采集数据，训练的非常慢，这也是PPO算法要改进的。

## 4、损失函数的优化

这个函数还有没有可以优化的地方呢，答案是有的，原式子中，我们的一个动作可以影响整个轨迹的回报的好坏，事实上我们的一个动作只能影响这个动作之后的轨迹的回报，而不会影响这个动作之前的奖励。我们将$R(\tau^n)$改为*$R_t^n$：从当前时间步$t$到结束的奖励

还有就是我们可以加入基线，让相对好的动作概率增加，让相对差的概率减少

$$
\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} (R_t^n - B(s_n^t)) \nabla \log P_\theta(a_n^t | s_n^t)
$$

### 公式组成部分

1. **$N$**：表示采样的轨迹数量。
2. **$T_n$**：表示第 $n$ 个轨迹的长度。
3. **$R_t^n$**：表示从当前时间步 $t$ 到结束的奖励。
4. **$B(s_n^t)$**：表示基线函数，通常是状态 $s_n^t$ 的价值函数 $V(s)$ 或优势函数 $A(s, a)$ 的估计。基线用于减少梯度估计的方差。
5. **$\nabla \log P_\theta(a_n^t | s_n^t)$**：表示在状态 $s_n^t$ 下采取动作 $a_n^t$ 的对数概率关于策略参数 $\theta$ 的梯度。这是策略梯度的一部分，用于指导策略参数的更新。

### 公式解释

- **外层求和 $\sum_{n=1}^{N}$**：对所有采样的轨迹进行求和。
- **内层求和 $\sum_{t=1}^{T_n}$**：对每个轨迹中的所有时间步进行求和。
- **$(R_t^n - B(s_n^t))$**：计算奖励与基线的差值，这个差值表示了实际奖励与预期奖励之间的偏差。这个偏差用于调整策略，使其倾向于选择那些能够获得更高奖励的动作。
- **$\nabla \log P_\theta(a_n^t | s_n^t)$**：计算对数概率的梯度，这个梯度指示了如何调整策略参数以增加在给定状态下选择特定动作的概率。

$R_t^n - B(s_n^t)$可以表示为优势函数，我们可以使用优势函数来替换基线的：

$$
\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} (A_{\theta}(s_n^t,a_n^t)) \nabla \log P_\theta(a_n^t | s_n^t)
$$

### 更新策略参数

这个梯度表达式通常用于梯度上升算法中，以更新策略参数 $\theta$：

$$
\theta \leftarrow \theta + \alpha \cdot \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} (A_{\theta}(s_n^t,a_n^t)) \nabla \log P_\theta(a_n^t | s_n^t)
$$

其中，$\alpha$ 是学习率，控制着参数更新的步长。

## 5. 广义优势估计

这个式子描述的是广义优势估计（Generalized Advantage Estimation，简称GAE）的计算公式。GAE是一种用于强化学习中策略梯度方法的算法，它通过结合多步估计来提高优势函数（Advantage Function）的估计效率和稳定性。

### GAE公式解释

$$
A_{\theta}^{GAE}(s_t, a) = (1 - \lambda)(A_{\theta}^1 + \lambda \cdot A_{\theta}^2 + \lambda^2 \cdot A_{\theta}^3 + \cdots)
$$

- $A_{\theta}^{GAE}(s_t, a)$：表示在状态 $s_t$ 下采取动作 $a$ 的广义优势估计值。
- $\lambda$：是一个介于0和1之间的超参数，用于控制多步估计的权重。较小的 $\lambda$ 值意味着更多地关注单步估计，而较大的 $\lambda$ 值意味着更多地关注多步估计。
- $A_{\theta}^k$：表示 $k$ 步优势估计，即从当前状态 $s_t$ 开始，考虑 $k$ 步未来奖励与价值函数的差值。

### 优势函数

优势函数 $A_{\theta}(s_t, a_t)$ 定义为采取动作 $a_t$ 相对于平均策略的相对优势，它可以通过以下方式计算：

$$
A_{\theta}(s_t, a_t) = Q_{\theta}(s_t, a_t) - V_{\theta}(s_t)
$$

其中：

- $Q_{\theta}(s_t, a_t)$ 是动作价值函数（Action-Value Function），表示在状态 $s_t$ 下采取动作 $a_t$ 并遵循策略 $\theta$ 后的期望回报。
- $V_{\theta}(s_t)$ 是状态价值函数（State-Value Function），表示在状态 $s_t$ 下遵循策略 $\theta$ 后的期望回报。

### 多步优势估计

多步优势估计 $A_{\theta}^k$ 考虑了从当前状态开始的 $k$ 步未来奖励与价值函数的差值，可以表示为：

$$
A_{\theta}^k = \sum_{l=1}^{k} \delta_l^{TD} \cdot \gamma^{l-1}
$$

其中：

- $\delta_l^{TD}$ 是时间差分误差（Temporal Difference Error），定义为 $r_{t+l} + \gamma V_{\theta}(s_{t+l}) - V_{\theta}(s_t)$，其中 $r_{t+l}$ 是在时间步 $t+l$ 获得的奖励，$\gamma$ 是折扣因子。
- $\gamma^{l-1}$ 是折扣因子的 $l-1$ 次幂，用于调整未来奖励的权重。

图中展示的是广义优势估计（GAE）的推导过程。GAE 是一种用于强化学习中策略梯度方法的算法，它通过结合多步估计来提高优势函数的估计效率和稳定性。下面是图中推导过程的详细解释：

推导过程：

1. **GAE 的定义**：
   $$
   A_{\theta}^{GAE}(s_t, a) = (1 - \lambda)(A_{\theta}^1 + \lambda A_{\theta}^2 + \lambda^2 A_{\theta}^3 + \cdots)
   $$
   这里，$A_{\theta}^k$ 表示 $k$ 步优势估计，$\lambda$ 是一个介于0和1之间的超参数，用于控制多步估计的权重。

2. **展开 $A_{\theta}^k$**：
   $$
   A_{\theta}^k = \delta_t^V + \gamma \delta_{t+1}^V + \gamma^2 \delta_{t+2}^V + \cdots + \gamma^{k-1} \delta_{t+k-1}^V
   $$
   其中，$\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是时间差分误差，$\gamma$ 是折扣因子。

3. **代入 $A_{\theta}^k$**：
   $$
   A_{\theta}^{GAE}(s_t, a) = (1 - \lambda)(\delta_t^V + \lambda(\delta_t^V + \gamma \delta_{t+1}^V) + \lambda^2(\delta_t^V + \gamma \delta_{t+1}^V + \gamma^2 \delta_{t+2}^V) + \cdots)
   $$

4. **整理**：
   $$
   A_{\theta}^{GAE}(s_t, a) = (1 - \lambda)(\delta_t^V (1 + \lambda + \lambda^2 + \cdots) + \gamma \delta_{t+1}^V (\lambda + \lambda^2 + \cdots) + \cdots)
   $$

5. **利用等比数列求和公式**：
   $$
   A_{\theta}^{GAE}(s_t, a) = (1 - \lambda)(\delta_t^V \frac{1}{1 - \lambda} + \gamma \delta_{t+1}^V \frac{\lambda}{1 - \lambda} + \cdots)
   $$

6. **简化**：
   $$
   A_{\theta}^{GAE}(s_t, a) = \sum_{b=0}^{\infty} (\gamma \lambda)^b \delta_{t+b}^V
   $$

这个推导过程展示了如何从GAE的定义出发，通过展开多步优势估计并利用等比数列求和公式，最终得到一个简洁的表达式。这个表达式表明，GAE可以看作是所有未来时间步的时间差分误差的加权和，其中权重由 $\gamma$ 和 $\lambda$ 决定。这种方法在强化学习中被广泛应用，因为它能够在不同时间尺度上平衡估计的偏差和方差。

## 6、最后推导出来的式子

图片中的内容是强化学习中计算广义优势估计（Generalized Advantage Estimation，简称GAE）的公式和损失函数的表达式。GAE是一种用于策略梯度方法中的优势函数估计技术，它通过结合多步估计来提高估计的效率和稳定性。下面是图片中各个部分的详细解释：

1. **时间差分误差（Temporal Difference Error）**:
   $$
   \delta_t^V = r_t + \gamma \cdot V_\theta(s_{t+1}) - V_\theta(s_t)
   $$
   这里，$r_t$ 是在时间步 $t$ 获得的奖励，$\gamma$ 是折扣因子，$V_\theta(s)$ 是状态 $s$ 的价值函数，参数为 $\theta$。

2. **广义优势估计（GAE）**:
   $$
   A_{\theta}^{GAE}(s_t, a) = \sum_{b=0}^{\infty} (\gamma \lambda)^b \delta_{t+b}^V
   $$
   这个公式表示GAE是所有未来时间步的时间差分误差的加权和，其中权重由 $\gamma$ 和 $\lambda$ 决定。$\lambda$ 是一个介于0和1之间的超参数，用于控制多步估计的权重。

3. **损失函数**:
   $$
   \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_{\theta}^{GAE}(s_n^t, a_n^t) \nabla \log P_\theta(a_n^t | s_n^t)
   $$
   这个公式是用于策略梯度方法中的损失函数，其中：
   - $N$ 是采样的轨迹数量。
   - $T_n$ 是第 $n$ 个轨迹的长度。
   - $A_{\theta}^{GAE}(s_n^t, a_n^t)$ 是在状态 $s_n^t$ 下采取动作 $a_n^t$ 的广义优势估计值。
   - $\nabla \log P_\theta(a_n^t | s_n^t)$ 是在状态 $s_n^t$ 下采取动作 $a_n^t$ 的对数概率关于策略参数 $\theta$ 的梯度。

这个损失函数用于梯度下降或上升算法中，以更新策略参数 $\theta$，从而提高策略的性能。通过最小化这个损失函数，我们可以有效地优化策略参数，使智能体在环境中的表现更好。

## 3. **PPO算法的实现步骤（off policy）**

off policy，它可以使用与当前正在学习的策略不同的策略生成的数据来学习。这种算法的优势在于它能够利用旧的数据进行学习，从而提高数据的利用效率。例如，Q-learning和DQN（Deep Q-Networks）是典型的off-policy算法。Off-policy算法的一个关键特点是它们允许从历史数据中学习，这些数据可能是由不同的探索策略产生的，与当前正在优化的策略并不一定相同。这种方法可以在不与环境进行实时互动的情况下，通过经验回放（Experience Replay）技术来训练智能体
PPO算法的实现通常包括以下几个步骤：

### （1）**重要性采样（Importance Sampling）**

重要性采样是一种统计方法，用于估计一个分布下的期望值，当直接从该分布中采样困难或不可行时，可以从一个近似分布中采样，然后通过重要性权重进行调整。

### 推导过程

1. **期望的定义**：
   $$
   \mathbb{E}(f(x))_{x \sim p(x)} = \sum_x f(x) \cdot p(x)
   $$
   这里，$\mathbb{E}(f(x))_{x \sim p(x)}$ 表示函数 $f(x)$ 在分布 $p(x)$ 下的期望值。

2. **引入另一个分布 $q(x)$**：
   $$
   = \sum_x f(x) \cdot p(x) \frac{q(x)}{q(x)}
   $$
   这里，我们在 $p(x)$ 前乘以 $q(x)/q(x)$，这样做的目的是为了能够将 $p(x)$ 转换为 $q(x)$。

3. **重写求和表达式**：
   $$
   = \sum_x f(x) \frac{p(x)}{q(x)} \cdot q(x)
   $$
   这一步将 $p(x)$ 和 $q(x)$ 分开，为后续的期望转换做准备。

4. **转换为 $q(x)$ 下的期望**：
   $$
   = \mathbb{E}\left(f(x) \frac{p(x)}{q(x)}\right)_{x \sim q(x)}
   $$
   这一步表明，我们现在可以在分布 $q(x)$ 下计算 $f(x) \frac{p(x)}{q(x)}$ 的期望值。

5. **近似估计**：
   $$
   \approx \frac{1}{N} \sum_{n=1}^N f(x) \frac{p(x)}{q(x)}_{x \sim q(x)}
   $$
   当直接计算 $q(x)$ 下的期望不可行时，我们可以通过从 $q(x)$ 中采样 $N$ 个样本，并计算这些样本的平均值来近似期望值。

重要性采样通过从一个易于采样的分布 $q(x)$ 中采样，并使用权重 $\frac{p(x)}{q(x)}$ 来调整样本，从而估计目标分布 $p(x)$ 下的期望值。这种方法在机器学习和统计学中非常有用，特别是在处理复杂分布或稀有事件时。

1. **给损失函数增加重要性采样**：

图片中的内容涉及强化学习中策略梯度方法的一个重要概念，即如何利用重要性采样（Importance Sampling）来调整策略梯度估计。这通常出现在off-policy学习场景中，即学习策略（target policy）与生成数据的策略（behavior policy）不一致时。

### 原始公式

$$
\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_{\theta}^{GAE}(s_n^t, a_n^t) \nabla \log P_{\theta}(a_n^t | s_n^t)
$$

这个公式表示的是策略梯度的估计，其中：

- $N$ 是采样的轨迹数量。
- $T_n$ 是第 $n$ 个轨迹的长度。
- $A_{\theta}^{GAE}(s_n^t, a_n^t)$ 是在状态 $s_n^t$ 下采取动作 $a_n^t$ 的广义优势估计值。
- $\nabla \log P_{\theta}(a_n^t | s_n^t)$ 是在状态 $s_n^t$ 下采取动作 $a_n^t$ 的对数概率关于策略参数 $\theta$ 的梯度。

### 增加重要性采样的损失函数公式

图片展示了强化学习中使用梯度上升方法来优化策略参数的损失函数的推导过程。这个过程涉及到广义优势估计（Generalized Advantage Estimation, GAE）和重要性采样权重的使用。以下是详细的步骤解释：

1. **初始损失函数**：
   $$
   \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_{\theta}^{GAE}(s_n^t, a_n^t) \nabla \log P_{\theta}(a_n^t | s_n^t)
   $$
   这里，$N$ 是采样的轨迹数量，$T_n$ 是第 $n$ 个轨迹的长度，$A_{\theta}^{GAE}(s_n^t, a_n^t)$ 是在状态 $s_n^t$ 下采取动作 $a_n^t$ 的广义优势估计值，$\nabla \log P_{\theta}(a_n^t | s_n^t)$ 是在状态 $s_n^t$ 下采取动作 $a_n^t$ 的对数概率关于策略参数 $\theta$ 的梯度。

2. **引入重要性采样权重**：
   $$
   = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_{\theta'}^{GAE}(s_n^t, a_n^t) \frac{P_{\theta}(a_n^t | s_n^t)}{P_{\theta'}(a_n^t | s_n^t)} \nabla \log P_{\theta}(a_n^t | s_n^t)
   $$
   这里，$P_{\theta}(a_n^t | s_n^t)$ 是目标策略在状态 $s_n^t$ 下采取动作 $a_n^t$ 的概率，$P_{\theta'}(a_n^t | s_n^t)$ 是行为策略在状态 $s_n^t$ 下采取动作 $a_n^t$ 的概率。重要性采样权重用于调整由于策略变化导致的概率差异。

3. **利用梯度和对数的关系**：
   $$
   = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_{\theta'}^{GAE}(s_n^t, a_n^t) \frac{P_{\theta}(a_n^t | s_n^t)}{P_{\theta'}(a_n^t | s_n^t)} \frac{\nabla P_{\theta}(a_n^t | s_n^t)}{P_{\theta}(a_n^t | s_n^t)}
   $$
   这里，利用了 $\nabla \log f(x) = \frac{\nabla f(x)}{f(x)}$ 的关系。

4. **简化表达式**：
   $$
   = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_{\theta'}^{GAE}(s_n^t, a_n^t) \frac{\nabla P_{\theta}(a_n^t | s_n^t)}{P_{\theta'}(a_n^t | s_n^t)}
   $$

5. **最终损失函数**：
   $$
   \text{Loss} = -\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_{\theta'}^{GAE}(s_n^t, a_n^t) \frac{P_{\theta}(a_n^t | s_n^t)}{P_{\theta'}(a_n^t | s_n^t)}
   $$
   最终的损失函数用于策略梯度方法中的梯度上升更新。通过最小化这个损失函数，我们可以优化策略参数 $\theta$，使智能体在环境中的表现更好。
6. **约束模型不要偏差太大**

图片中展示了两种不同的损失函数，它们用于近端策略优化（Proximal Policy Optimization，简称PPO）算法中。PPO是一种强化学习算法，它通过限制策略更新的幅度来提高策略梯度方法的稳定性。以下是图片中两个损失函数的详细解释：

### 第一个损失函数（PPO1）（使用KL散度）

$$
Loss_{PPO1} = -\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_{\theta'}^{GAE}(s_n^t, a_n^t) \frac{P_{\theta}(a_n^t | s_n^t)}{P_{\theta'}(a_n^t | s_n^t)} + \beta KL(P_{\theta}, P_{\theta'})
$$

- $N$ 是采样的轨迹数量。
- $T_n$ 是第 $n$ 个轨迹的长度。
- $A_{\theta'}^{GAE}(s_n^t, a_n^t)$ 是在状态 $s_n^t$ 下采取动作 $a_n^t$ 的广义优势估计值，基于旧策略参数 $\theta'$ 计算。
- $P_{\theta}(a_n^t | s_n^t)$ 和 $P_{\theta'}(a_n^t | s_n^t)$ 分别是新旧策略在状态 $s_n^t$ 下采取动作 $a_n^t$ 的概率。
- $\beta KL(P_{\theta}, P_{\theta'})$ 是新旧策略之间的Kullback-Leibler散度，用于惩罚新策略与旧策略之间的差异过大，$\beta$ 是一个超参数，用于控制这种惩罚的强度。

### 第二个损失函数（PPO2）（使用裁剪）

$$
Loss_{PPO2} = -\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} \min \left( A_{\theta'}^{GAE}(s_n^t, a_n^t) \frac{P_{\theta}(a_n^t | s_n^t)}{P_{\theta'}(a_n^t | s_n^t)}, \text{clip} \left( \frac{P_{\theta}(a_n^t | s_n^t)}{P_{\theta'}(a_n^t | s_n^t)}, 1 - \epsilon, 1 + \epsilon \right) A_{\theta'}^{GAE}(s_n^t, a_n^t) \right)
$$

- 这个损失函数使用了裁剪（clip）操作来限制策略更新的幅度。裁剪函数 $\text{clip}(x, 1 - \epsilon, 1 + \epsilon)$ 确保比率 $\frac{P_{\theta}(a_n^t | s_n^t)}{P_{\theta'}(a_n^t | s_n^t)}$ 被限制在 $1 - \epsilon$ 和 $1 + \epsilon$ 之间，其中 $\epsilon$ 是一个超参数，通常取较小的值（如0.1或0.2）。

### 总结

PPO算法通过这两种损失函数来平衡策略的学习效率和更新稳定性。第一种损失函数通过KL散度来惩罚策略变化过大，而第二种损失函数通过裁剪操作直接限制策略更新的幅度。这两种方法都是为了确保策略更新不会过于激进，从而提高算法的稳定性和可靠性。

总结来说，引入重要性采样权重和利用梯度与对数的关系来推导和优化策略梯度方法中的损失函数。这种方法允许我们在off-policy学习场景中有效地利用旧数据进行学习。off-policy强化学习中使用重要性采样来调整策略梯度估计，以便利用与目标策略不同的行为策略生成的数据进行学习。**我们可以用使用对另一个模型的评价来更新模型**。
