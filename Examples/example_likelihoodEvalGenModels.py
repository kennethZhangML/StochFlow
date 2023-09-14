from COV_DetFlow.GaussianMixtureModel import *


def generate_samples(model, num_samples):
    components = len(model.means)
    components_idx = np.random.choice(components, size=num_samples, p=model.weights)
    samples = np.zeros((num_samples, model.means[0].shape[0]))

    for i in range(components):
        num_samples_i = np.sum(components_idx == i)
        samples[components_idx == i] = np.random.multivariate_normal(
            model.means[i], model.covariances[i], num_samples_i)

    return samples


if __name__ == "__main__":
    initial_means = np.array([[-1, -1], [1, 1]])
    initial_covariances = [np.array([[0.1, 0.05], [0.05, 0.1]]),
                           np.array([[0.1, -0.05], [-0.05, 0.1]])]
    initial_weights = [0.5, 0.5]
    rho_0 = GaussianMixtureModel(initial_means, initial_covariances, initial_weights)

    final_means = np.array([[1, 1], [-1, -1]])
    final_covariances = [np.array([[0.1, 0.05], [0.05, 0.1]]),
                         np.array([[0.1, -0.05], [-0.05, 0.1]])]
    final_weights = [0.5, 0.5]
    rho_1 = GaussianMixtureModel(final_means, final_covariances, final_weights)

    samples_forward = generate_samples(rho_1, num_samples=1000)


    def prob_density_rho_0(x):
        return np.sum([w * multivariate_normal.pdf(x, mean=mu, cov=cov) for w, mu, cov in
                       zip(rho_0.weights, rho_0.means, rho_0.covariances)])


    def prob_density_rho_1(x):
        return np.sum([w * multivariate_normal.pdf(x, mean=mu, cov=cov) for w, mu, cov in
                       zip(rho_1.weights, rho_1.means, rho_1.covariances)])


    likelihood_forward = likelihood(prob_density_rho_1, samples_forward)
    cross_entropy_estimate = cross_entropy(prob_density_rho_0, prob_density_rho_1, samples_forward)

    print(f"Likelihood of forward generative model: {likelihood_forward}")
    print(f"Cross-entropy between rho_0 and rho_1: {cross_entropy_estimate}")
