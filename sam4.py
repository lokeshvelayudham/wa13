import numpy as np
import matplotlib.pyplot as plt

# Define the likelihoods for h3 and h4
likelihood_h3 = 0.5
likelihood_h4 = 0.75

# Prior probabilities for h3 and h4
prior_h3 = 0.9
prior_h4 = 0.1

# Function to calculate posterior probability
def posterior(likelihood, prior, N):
    # Since we are observing lime candies only, the data is the same for all N
    data_likelihood = likelihood ** N
    return data_likelihood * prior

# Since we want separate plots for h3 and h4, we'll need to recompute the probabilities where we don't switch the hypothesis

# Function to calculate maximum a posteriori probability (hMAP) for h3 and h4
def calculate_hMAP(likelihood_h3, likelihood_h4, prior_h3, prior_h4, N):
    post_h3 = posterior(likelihood_h3, prior_h3, N)
    post_h4 = posterior(likelihood_h4, prior_h4, N)
    norm_constant = post_h3 + post_h4
    post_h3_normalized = post_h3 / norm_constant
    post_h4_normalized = post_h4 / norm_constant
    return post_h3_normalized, post_h4_normalized

# Calculate hMAP for h3 and h4 separately
prob_lime_h3_MAP = []
prob_lime_h4_MAP = []

for N in range(1, 101):
    post_h3, post_h4 = calculate_hMAP(likelihood_h3, likelihood_h4, prior_h3, prior_h4, N)
    prob_lime_h3_MAP.append(likelihood_h3 if post_h3 > post_h4 else 0)  # Only add if h3 is MAP
    prob_lime_h4_MAP.append(likelihood_h4 if post_h4 > post_h3 else 0)  # Only add if h4 is MAP

# Now we plot the probabilities for h3 and h4 separately
plt.figure(figsize=(14, 7))

# Plot for h3
plt.subplot(1, 2, 1)
plt.plot(range(1, 101), prob_lime_h3_MAP, label='P(DN+1=lime|h3 MAP)', color='green')
plt.xlabel('Number of observations in d (N)')
plt.ylabel('Probability that next candy is lime')
plt.title('Probability of Lime Candy Given h3 MAP')
plt.legend()
plt.grid(True)

# Plot for h4
plt.subplot(1, 2, 2)
plt.plot(range(1, 101), prob_lime_h4_MAP, label='P(DN+1=lime|h4 MAP)', color='purple')
plt.xlabel('Number of observations in d (N)')
plt.title('Probability of Lime Candy Given h4 MAP')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
