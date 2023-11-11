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

# Arrays to hold the probability values
prob_lime_hMAP = []
prob_lime_hML = []

# Calculate probabilities for N ranging from 1 to 100
for N in range(1, 101):
    # Calculate posteriors for both hypotheses
    post_h3 = posterior(likelihood_h3, prior_h3, N)
    post_h4 = posterior(likelihood_h4, prior_h4, N)
    
    # Normalize to get actual posterior probabilities
    norm_constant = post_h3 + post_h4
    post_h3 /= norm_constant
    post_h4 /= norm_constant
    
    # hMAP is the hypothesis with the highest posterior probability
    if post_h3 > post_h4:
        prob_lime_hMAP.append(likelihood_h3)  # Probability of lime for h3
    else:
        prob_lime_hMAP.append(likelihood_h4)  # Probability of lime for h4
    
    # hML is the hypothesis with the maximum likelihood given the data
    # Since h4 always has a higher likelihood for lime, it will be hML
    prob_lime_hML.append(likelihood_h4)  # Probability of lime for h4

# Now plot the probabilities
N_values = range(1, 101)
plt.figure(figsize=(14, 7))

# Plot P(DN+1=lime|hMAP)
plt.plot(N_values, prob_lime_hMAP, label='P(DN+1=lime|hMAP)', color='blue')

# Plot P(DN+1=lime|hML)
plt.plot(N_values, prob_lime_hML, label='P(DN+1=lime|hML)', color='red', linestyle='--')

plt.xlabel('Number of observations in d (N)')
plt.ylabel('Probability that next candy is lime')
plt.title('Predicted Probability of Lime Candy')
plt.legend()
plt.grid(True)
plt.show()
