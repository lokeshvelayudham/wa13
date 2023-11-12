import numpy as np
import matplotlib.pyplot as plt

# Hypotheses
cherry_hypo = [1, 0.75, 0.5, 0.25, 0]
lime_hypo = [0, 0.25, 0.5, 0.75, 1]

# Prior probabilities for each hypothesis
priors = np.array([0.1, 0.2, 0.4, 0.2, 0.1])


def generate_data(hypothesis, N=100):
    return np.random.choice(['cherry', 'lime'], N, p=[cherry_hypo[hypothesis], lime_hypo[hypothesis]])


def calculate_posterior(data):
    likelihoods = []
    for h in range(5):
        likelihood = np.prod(
            [cherry_hypo[h] if d == 'cherry' else lime_hypo[h] for d in data])
        likelihoods.append(likelihood)

    evidence = sum(np.array(likelihoods) * np.array(priors))
    posteriors = (np.array(likelihoods) * np.array(priors)) / evidence
    # print("old:", posteriors)
    return posteriors


def predict_next_candy(data):
    posteriors = calculate_posterior(data)
    return sum(posteriors[i] * lime_hypo[i] for i in range(5))


# to plot the graph
# For hypothesis h3 and h4:
datasets = {f"h{i}": generate_data(i-1) for i in [3, 4]}

def find_hmap_hml(posteriors, prior_h3=0.9):
    # Adjust priors for h3 and h4
    adjusted_priors = [0, 0, prior_h3, 1 - prior_h3, 0]

    # hMAP: hypothesis with the highest posterior probability
    normalized_posterior = [posteriors[i]/1 for i in range(5)]
    hmapcal = [normalized_posterior[i]*adjusted_priors[i] for i in range(5)]
    hmap = max(hmapcal)
    # calculate the maximum value and then return the index of max
    hmapind = hmapcal.index(hmap)

    # print(posteriors)

    # hML: among h3 and h4, the one with the higher adjusted posterior
    hmlmax = max(normalized_posterior)

   # calculate the maximum value and then return the index of max
    hmlindex = normalized_posterior.index(hmlmax)

    return hmapind, hmlindex


# plot
for hypothesis, data in datasets.items():
    posteriors_over_time = []
    predictions_over_time = []
    predictions_hmap_over_time = []
    predictions_hml_over_time = []

    for N in range(0, 100):
        current_data = data[:N]
        posteriors = calculate_posterior(current_data)
        posteriors_over_time.append(posteriors)
        predictions_over_time.append(predict_next_candy(current_data))
        hmap, hml = find_hmap_hml(posteriors)
        predictions_hmap_over_time.append(lime_hypo[hmap])
        predictions_hml_over_time.append(lime_hypo[hml])

    # Plotting Posterior Probabilities and Predictions for Lime Candy
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for i in range(5):
        plt.plot(range(0, 100), [posterior[i]for posterior in posteriors_over_time], label=f'h{i+1}')
    plt.xlabel('N')
    plt.ylabel('P(hi|d)')
    plt.legend()
    plt.title(f'Posterior Probabilities for {hypothesis}')
    plt.subplot(1, 2, 2)
    plt.plot(range(0, 100), predictions_over_time, label='P(DN+1=lime|d)')
    plt.plot(range(0, 100), predictions_hmap_over_time,label='P(DN+1=lime|hMAP)', linestyle='--')
    plt.plot(range(0, 100), predictions_hml_over_time, label='P(DN+1=lime|hML)', linestyle='-.')
    plt.xlabel('N')
    plt.ylabel('Probability')
    plt.title(f'Prediction for Lime Candy for {hypothesis}')
    plt.legend()
    plt.tight_layout()
    plt.show()