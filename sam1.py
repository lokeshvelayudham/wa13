import numpy as np
import matplotlib.pyplot as plt

# Probabilities associated with each hypothesis
p_h = [0.2, 0.4, 0.6, 0.8, 1.0]
priors = [0.2, 0.2, 0.2, 0.2, 0.2]

def generate_data(p, N=100):
    return np.random.choice([0, 1], size=N, p=[1-p, p])

def update_posteriors(data):
    all_posteriors = []
    for N in range(1, len(data)+1):
        likelihoods = [np.prod([(p**d) * ((1-p)**(1-d)) for d in data[:N]]) for p in p_h]
        evidence = sum([likelihoods[i] * priors[i] for i in range(5)])
        posteriors = [likelihoods[i] * priors[i] / evidence for i in range(5)]
        all_posteriors.append(posteriors)
    return all_posteriors

def predict_next(all_posteriors):
    predictions = []
    for posteriors in all_posteriors:
        prediction = sum([p_h[i] * posteriors[i] for i in range(5)])
        predictions.append(prediction)
    return predictions

# Simulate data for h3 and h4
data_h3 = generate_data(0.5)
data_h4 = generate_data(0.75)

# Update posteriors
all_posteriors_h3 = update_posteriors(data_h3)
all_posteriors_h4 = update_posteriors(data_h4)

# Predict next data point
predictions_h3 = predict_next(all_posteriors_h3)
predictions_h4 = predict_next(all_posteriors_h4)

# Plot predictions for h3 and h4
plt.figure(figsize=(8, 6))
plt.plot(predictions_h3, label="h3 (50% cherry + 50% lime)", linestyle='--')
plt.plot(predictions_h4, label="h4 (25% cherry + 75% lime)", linestyle='-')
plt.title("Bayesian Prediction \( P(D_{N+1} = \text{lime} | d_1, …, d_N) \) for h3 and h4")
plt.xlabel("Number of samples in d")
plt.ylabel("Probability \( P(D_{N+1} = \text{lime}) \)")
plt.legend()
plt.show()
 