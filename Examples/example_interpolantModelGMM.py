from COV_DetFlow.StochasticInterpolantModel import * 

if __name__ == "__main__":
    means_0 = [-2, 2]
    covariances_0 = [0.5, 0.3]
    weights_0 = [0.4, 0.6]
    rho_0 = GaussianMixtureModel(means_0, covariances_0, weights_0)

    means_1 = [0, 3]
    covariances_1 = [0.2, 0.4]
    weights_1 = [0.6, 0.4]
    rho_1 = GaussianMixtureModel(means_1, covariances_1, weights_1)

    params = {
        'interpolant': 'linear',
        'diffusivity': 'constant',
        'interpolant_params': {'alpha': 0.7},  
        'diffusivity_params': {'D': 0.1}  
    }

    model = StochasticInterpolantModel(rho_0, rho_1, params)
    samples = model.generate_samples(num_samples = 1000)

    likelihood = model.likelihood(samples)
    cross_entropy = model.cross_entropy(samples)

    print('Likelihood of SIM for Gaussian Mixture: ', likelihood)
    print('Cross-Entropy of SIM for Gaussian Mixture: ', cross_entropy)