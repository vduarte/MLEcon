# Copyright 2018 Victor Duarte. All Rights Reserved.
import mlecon as mle
import matplotlib.pyplot as plt
import numpy as np
# %% ----------------- Setup ------------------------------------
mle.clear()
dt = mle.dt
Δt = 0.1
n_trees = 2

# Hyper-parameters
hidden = [64, 32, 8]  # number of units in each hidden layer

# State space
D = mle.states(2)  # Create 2 state variables (dividends)

# Shocks
dZ = mle.brownian_shocks(n_trees)

# Bounds
bounds = {D[i]: [1e-4, 100] for i in range(n_trees)}
sample = mle.sampler(bounds)  # Op that performs the sampling

# Function approximator
inputs = [mle.log(D[i]) for i in range(n_trees)]
J = mle.network(inputs, name='VF', hidden=hidden)

# %% ------------ Economic Model -------------------------------
δ = 0.04
σ = [0.2, -.3]
μ = [.02, .03]
ρ = -0.5

C = sum(D)               # Aggregate consumption
M = C**(-1)              # Marginal Utility
P = J / M                # Price of the first tree
U = D[0] * M             # div x marg. utility
d1 = D[0] / P            # dividend yield for the first tree

# Environment dynamics
D[0].d = μ[0] * D[0] * dt + σ[0] * D[0] * dZ[0]
D[1].d = μ[1] * D[1] * dt + ρ * σ[1] * D[1] * dZ[0] +\
    np.sqrt(1 - ρ**2) * σ[1] * D[1] * dZ[1]

HJB = U - δ * J + J.drift  # Bellman residual
T = J + Δt * HJB           # Bellman target
regress = mle.fit(J, T)  # Update the ANN J to fit the target T using SGD

Er = P.drift / P + d1  # Expected returns
variance = P.var / P**2  # returns variance
r = -M.drift / M + δ     # riskfree rate
s = D[0] / C


# Launch graph
env = mle.environment(J)


# %% --- Test function ------------------------------------------------
def test():
    s_, d_ = env([s, d1])
    plt.scatter(s_, d_, s=1, color=plt.cm.Blues(.7))
    env.show()


# %% --- Iteration ----------------------------------------------------
program = {sample: 1, regress: 1, test: 500}
env.iterate(program, T='00:00:30')  # Iterate for 30 s
