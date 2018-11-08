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
batch_size = 256
mle.set_batch_size(batch_size)

# State variables
K = mle.state(.1, 10)
z = mle.state(-0.5, .5)
σ = mle.state(-6, -1)

# Brownian Shocks
dZ = mle.brownian_shocks(2)

# Model parameters
β, ν, ζ, δ, λ, σ_, γ, η, ρ, ψ = 0.04, 0.36, .3, 0.0196, 0.95, -3, 4, .1, .9, .5
θ = (1 - γ) / (1 - 1 / ψ)

# Function approximators
J = mle.network([K, z, σ], hidden, name='Value_function')
L = mle.network([K, z, σ], hidden, name='labor', bnds=[1e-6, .999],
                activation_fn=tf.nn.relu)
s = mle.network([K, z, σ], hidden, name='savings', bnds=[1e-6, .999])

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
HJB = f + J.drift()  # Bellman residual
T = J + Δt * HJB   # Bellman target

policy_eval = mle.fit(J, T)
policy_improv = mle.greedy(HJB, actions=[s, L])

# %% ---- # Launch graph ---------------------------------
mle.launch()

# %% -----------  Create summaries for TensorBoard ----------
""" To see the summaries in real time during training open
a terminal and type:
tensorboard --logdir=./results/summaries
 """
scalars = {'HJB': tf.log(tf.reduce_mean(tf.abs(HJB(0)))),
           'MSE': tf.log(J.net.loss)}

mle.set_summary(scalars)


# %% -----------  Test function -----------------------------
feed_dict = {K: np.linspace(1, 10, batch_size),
             z: 0,
             σ: -3}


def test():
    K_, C_, L_ = mle.eval([K, C, L], feed_dict)

    plt.subplot(1, 2, 1)
    plt.plot(K_, L_)
    plt.xlabel('K')
    plt.ylabel('L')

    plt.subplot(1, 2, 2)
    plt.plot(K_, C_)
    plt.xlabel('K')
    plt.ylabel('C')

    plt.show()
    plt.pause(1e-6)


# %% -----------  Iteration --------------------------------------
# mle.load()  # Load previous results?
program = {policy_eval: 1,
           policy_improv: 1,
           test: 1000,
           mle.add_summary: 100}

mle.iterate(program, T='01:00:00')
mle.save()
