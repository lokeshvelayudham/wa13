import numpy as np
import matplotlib.pyplot as plt

# Hypotheses
p_cherry_given_h = [1, 0.75, 0.5, 0.25, 0]
p_lime_given_h = [0, 0.25, 0.5, 0.75, 1]

# Prior probabilities for each hypothesis
p_h = [1/5]*5

def generate_data(hypothesis, N=100):
    return np.random.choice(['cherry', 'lime'], N, p=[p_cherry_given_h[hypothesis], p_lime_given_h[hypothesis]])

def compute_posterior(data):
    likelihoods = []
    for h in range(5):
        likelihood = np.prod([p_cherry_given_h[h] if d == 'cherry' else p_lime_given_h[h] for d in data])
        likelihoods.append(likelihood)
    
    evidence = sum(np.array(likelihoods) * np.array(p_h))
    posteriors = (np.array(likelihoods) * np.array(p_h)) / evidence
    return posteriors

def predict_next_candy(data):
    posteriors = compute_posterior(data)
    return sum(posteriors[i] * p_lime_given_h[i] for i in range(5))

# For hypothesis h3 and h4:
datasets = {f"h{i}": generate_data(i-1) for i in [3, 4]}

for hypothesis, data in datasets.items():
    posteriors_over_time = []
    predictions_over_time = []

    for N in range(1, 101):
        posteriors_over_time.append(compute_posterior(data[:N]))
        predictions_over_time.append(predict_next_candy(data[:N]))

    plt.figure(figsize=(12, 5))

    # Plotting P(hi|d1,…dN)
    plt.subplot(1, 2, 1)
    for i in range(5):
        plt.plot(range(1, 101), [posterior[i] for posterior in posteriors_over_time], label=f'h{i+1}')
    plt.xlabel('N')
    plt.ylabel('P(hi|d)')
    plt.legend()
    plt.title(f'Posterior Probabilities for {hypothesis}')

    # Plotting P(DN+1=lime|d1,…dN)
    plt.subplot(1, 2, 2)
    plt.plot(range(1, 101), predictions_over_time, marker='.')
    plt.xlabel('N')
    plt.ylabel('P(DN+1=lime|d)')
    plt.title(f'Prediction for Lime Candy for {hypothesis}')

    plt.tight_layout()
    plt.show()
