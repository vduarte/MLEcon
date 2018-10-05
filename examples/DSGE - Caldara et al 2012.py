import mlecon as mle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Reset graph
mle.clear()

# Hyper-parameters
hidden = [32, 16, 8]
dt = mle.dt
Δt = 0.5
mle.delta_t = Δt
mle.batch_size = 128

K, z, σ = mle.states(3)  # State variables
dZ = mle.brownian_shocks(2)  # Brownian Shocks

# Model parameters
β, ν, ζ, δ, λ, σ_, γ, η, ρ, ψ = 0.04, 0.36, .3, 0.0196, 0.95, -3, 4, .1, .9, .5
θ = (1 - γ) / (1 - 1 / ψ)

# Bounds
bounds = {K: [.1, 10], z: [-.5, .5], σ: [-6, -1]}
sample = mle.sampler(bounds)

# Function approximators
inputs = [K, z, σ]

J = mle.network(inputs, hidden, name='Value_function')
L = mle.network(inputs, hidden, name='labor', bnds=[1e-6, .999],
                activation_fn=tf.nn.relu)
s = mle.network(inputs, hidden, name='savings', bnds=[1e-6, .999])

# %% -----------  Economic Model -----------------------------
Y = mle.exp(z) * K**ζ * L**(1 - ζ)
C = (1 - s) * Y
U = (C**ν * (1 - L)**(1 - ν))**(1 - γ) / (1 - γ)
I = s * Y

# Dynamics
K.d = (I - δ * K) * dt
z.d = -(1 - λ) * z * dt + mle.exp(σ) * dZ[0]
σ.d = (1 - ρ) * (σ_ - σ) * dt + η * dZ[1]

f = β * θ * J * ((mle.abs(U / J))**(1 / θ) - 1)  # Duffie-Epstein aggregator
HJB = f + J.drift  # Bellman residual
T = J + Δt * HJB   # Bellman target

policy_eval = mle.fit(J, T)
policy_improv = mle.greedy(HJB, actions=[s, L])

# %% ---- Stabilizing policy ---------------------------------
initial = tf.group(mle.fit(L, .3),
                   mle.fit(s, 1 - ρ))

# %% ---- # Launch graph ---------------------------------
env = mle.environment()

# %% -----------  Create summaries for TensorBoard ----------
""" To see the summaries in real time during training open
a terminal and type:
tensorboard --logdir=./results/summaries
 """
scalars = {'HJB': tf.log(tf.reduce_mean(tf.abs(HJB()))),
           'MSE': tf.log(J.net.loss)}

env.summary.set(scalars)


# %% -----------  Test function -----------------------------
def test():
    eval_dict = {z: 0, σ: -3}  # Reference points: z = 0 and σ = -3

    plt.figure(1, figsize=[9, 9])
    K_, C_, L_, J_ = env([K, C, L, J], eval_dict)
    idx_sort = np.argsort(K_.ravel())

    plt.subplot(1, 3, 1)
    plt.plot(K_[idx_sort], L_[idx_sort], color='b')
    plt.xlabel('K')
    plt.ylabel('L')

    plt.subplot(1, 3, 2)
    plt.plot(K_[idx_sort], C_[idx_sort], color='b')
    plt.xlabel('K')
    plt.ylabel('C')

    plt.subplot(1, 3, 3)
    plt.plot(K_[idx_sort], J_[idx_sort], color='b')
    plt.xlabel('K')
    plt.ylabel('J')

    env.show()


# %% ----------- Initialization ---------------------------------
program = {sample: 1, initial: 1, policy_eval: 1, test: 500}
env.iterate(program, T='00:00:10')

# %% -----------  Iteration --------------------------------------
# env.load()  # Load previous results?
program = {sample: 1,
           policy_eval: 1,
           policy_improv: 1,
           test: 1000,
           env.add_summary: 50}
env.iterate(program, T='01:00:00')
env.save()
