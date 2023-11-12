import matplotlib.pyplot as plt
import numpy as np

# Step 2: Set Prior Probability and Define Likelihood Functions
# Prior probability of h3
prior_h3 = 0.9

# Arbitrary likelihood functions for h3 and h4
def likelihood_h3(N):
    return np.exp(-0.1 * N)

def likelihood_h4(N):
    return np.exp(-0.05 * N)

# Step 3: Initialize Lists for Probabilities
# Initialize probabilities
prob_map = []
prob_ml = []

# Step 4: Calculate Probabilities for Each N
# Iterate over N from 1 to 100
for N in range(1, 101):
    # Calculate likelihoods
    lh_h3 = likelihood_h3(N)
    lh_h4 = likelihood_h4(N)

    # Calculate posterior probabilities
    posterior_h3 = (lh_h3 * prior_h3) / ((lh_h3 * prior_h3) + (lh_h4 * (1 - prior_h3)))
    posterior_h4 = 1 - posterior_h3

    # Update prior for the next iteration
    prior_h3 = posterior_h3

    # Probability of observing lime given h_MAP and h_ML
    prob_map.append(lh_h3 * posterior_h3 + lh_h4 * posterior_h4)
    prob_ml.append(max(lh_h3, lh_h4))

# Step 5: Plot the Results
# Plot the results
plt.plot(range(0, 100), prob_map, label='P(DN+1=lime|h_MAP)')
plt.plot(range(0, 100), prob_ml, label='P(DN+1=lime|h_ML)')
plt.xlabel('N')
plt.ylabel('Probability')
plt.legend()
plt.title('Probability of Observing "lime" under Different Hypotheses')
plt.show()