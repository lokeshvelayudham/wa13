import numpy as np
import matplotlib.pyplot as plt

# Define the prior probabilities and the likelihoods of observing a lime candy for each hypothesis
priors = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
likelihoods = np.array([[1.0, 0.0], [0.75, 0.25], [0.5, 0.5], [0.25, 0.75], [0.0, 1.0]])

# Function to simulate the data for a given hypothesis
def simulate_data(hypothesis, N):
    # hypothesis - 1 because hypothesis numbering starts at 1
    probabilities = likelihoods[hypothesis - 1]
    data = np.random.choice([0, 1], size=N, p=probabilities)
    return data

# Function to calculate the posterior probabilities
def calculate_posteriors(data, priors, likelihoods):
    posteriors = np.zeros((len(data) + 1, len(priors)))

    for i in range(len(data) + 1):
        if i == 0:
            # Before any data, the posteriors are just the priors
            posteriors[i] = priors
        else:
            # Calculate the likelihood of the data up to that point
            likelihood = likelihoods[:, data[i-1]]
            # Update the posteriors using Bayes' rule
            posteriors[i] = priors * likelihood
            # Normalize the posteriors
            posteriors[i] /= np.sum(posteriors[i])
            # Update priors for the next iteration
            priors = posteriors[i]

    return posteriors

# Function to calculate the predictive distribution for the next candy being lime
def calculate_predictive_distribution(posteriors, likelihoods):
    # Weigh the likelihood of lime by the posteriors
    predictive_distribution = posteriors @ likelihoods[:, 1]
    return predictive_distribution

# Simulate data for h3 and h4
data_h3 = simulate_data(3, 100)
data_h4 = simulate_data(4, 100)

# Calculate the posteriors for each set of data
posteriors_h3 = calculate_posteriors(data_h3, priors, likelihoods)
posteriors_h4 = calculate_posteriors(data_h4, priors, likelihoods)

# Calculate the predictive distribution for each set of data
predictive_distribution_h3 = calculate_predictive_distribution(posteriors_h3, likelihoods)
predictive_distribution_h4 = calculate_predictive_distribution(posteriors_h4, likelihoods)

# Plotting the posteriors for h3
for i in range(5):
    plt.plot(posteriors_h3[:, i], label=f'P(h{i+1}|d)')

plt.title('Posterior Probabilities for h3')
plt.xlabel('Number of Observations in d')
plt.ylabel('Posterior Probability of Hypothesis')
plt.legend()
plt.show()

# Plotting the posteriors for h4
for i in range(5):
    plt.plot(posteriors_h4[:, i], label=f'P(h{i+1}|d)')

plt.title('Posterior Probabilities for h4')
plt.xlabel('Number of Observations in d')
plt.ylabel('Posterior Probability of Hypothesis')
plt.legend()
plt.show()

# Plotting the predictive distributions
plt.figure(figsize=(14, 7))

# Predictive distribution for h3
plt.subplot(1, 2, 1)
plt.plot(predictive_distribution_h3, label='P(DN+1=lime|d) for h3')
plt.title('Predictive Probability for h3')
plt.xlabel('Number of Observations in d')
plt.ylabel('Probability that next candy is lime')
plt.legend()

# Predictive distribution for h4
plt.subplot(1, 2, 2)
plt.plot(predictive_distribution_h4, label='P(DN+1=lime|d) for h4')
plt.title('Predictive Probability for h4')
plt.xlabel('Number of Observations in d')
plt.ylabel('Probability that next candy is lime')
plt.legend()

plt.tight_layout()
plt.show()


# Define the new prior probabilities with h3 having a prior of 0.9
new_priors = np.array([0.025, 0.025, 0.9, 0.025, 0.025])

# Function to calculate MAP and ML probabilities
def calculate_map_ml(posteriors, likelihoods):
    # Maximum a posteriori (MAP) hypothesis
    h_map_index = np.argmax(posteriors, axis=1)
    # Maximum likelihood (ML) hypothesis is simply the likelihood of lime for the observed data
    h_ml_index = np.argmax(likelihoods[:, 1])

    # Calculate the predictive probabilities for MAP and ML
    p_lime_map = likelihoods[h_map_index, 1]
    p_lime_ml = likelihoods[h_ml_index, 1]

    return p_lime_map, p_lime_ml

# Recalculate the posteriors with the new priors
posteriors_h3_new_prior = calculate_posteriors(data_h3, new_priors, likelihoods)
posteriors_h4_new_prior = calculate_posteriors(data_h4, new_priors, likelihoods)

# Calculate the MAP and ML probabilities
p_lime_map_h3, p_lime_ml_h3 = calculate_map_ml(posteriors_h3_new_prior, likelihoods)
p_lime_map_h4, p_lime_ml_h4 = calculate_map_ml(posteriors_h4_new_prior, likelihoods)

# Plotting the MAP and ML probabilities
plt.figure(figsize=(14, 7))

# MAP and ML for h3
plt.subplot(1, 2, 1)
plt.plot(p_lime_map_h3, label='P(DN+1=lime|h_MAP) for h3',  color= 'blue',linestyle='--')
plt.plot(p_lime_ml_h3, label='P(DN+1=lime|h_ML) for h3', color= 'red', linestyle='-')
plt.title('Predictive Probability MAP and ML for h3')
plt.xlabel('Number of Observations in d')
plt.ylabel('Probability that next candy is lime')
plt.legend()

# MAP and ML for h4
plt.subplot(1, 2, 2)
plt.plot(p_lime_map_h4, label='P(DN+1=lime|h_MAP) for h4', linestyle='--')
plt.plot(p_lime_ml_h4, label='P(DN+1=lime|h_ML) for h4', linestyle='-')
plt.title('Predictive Probability MAP and ML for h4')
plt.xlabel('Number of Observations in d')
plt.ylabel('Probability that next candy is lime')
plt.legend()

plt.tight_layout()
plt.show()
