import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import quad
from scipy.integrate import solve_ivp
from sentence_transformers import util, SentenceTransformer
from Utils.util import interpolant_func, diffusivity_func


class TextStochasticInterpolantModel:
    def __init__(self, initial_model, final_model, time_interval, time_step):
        self.initial_model = initial_model
        self.final_model = final_model
        self.time_interval = time_interval
        self.time_step = time_step
        self.interpolant_func = interpolant_func
        self.diffusivity_func = diffusivity_func

    def generate_samples(self, num_samples):
        initial_samples = self.initial_model.encode(["initial sentence"] * num_samples)
        return self._integrate_samples(initial_samples)

    def _integrate_samples(self, samples):
        def time_derivative(t, y):
            interpolant = self.interpolant_func(t)
            diffusivity = self.diffusivity_func(t)
            dydt = interpolant(y) * diffusivity(y)
            return dydt

        samples_integrated = []
        for sample in samples:
            sol = solve_ivp(time_derivative, [0, self.time_interval],
                            sample, t_eval=[self.time_interval], method='RK45')
            # Unresolved attribute reference 'y' for class
            samples_integrated.append(sol.y[:, -1])
        return torch.tensor(samples_integrated)

    def likelihood(self, samples):
        return self.final_model.likelihood(samples)

    def cross_entropy(self):
        def integrand(x):
            initial_density_value = self.initial_model.likelihood(x.reshape(-1, 1))
            final_density_value = self.final_model.likelihood(x.reshape(-1, 1))
            return initial_density_value * np.log(initial_density_value / final_density_value)
        # numpy warning: Too many values to unpack
        integral_result, _ = quad(integrand, -np.inf, np.inf)
        return integral_result

    def visualize_sample_evolution(self, num_samples):
        initial_samples = self.initial_model.encode(["initial sentence"] * num_samples)
        time_points = np.arange(0, self.time_interval + self.time_step, self.time_step)

        sample_trajectories = []
        for sample in initial_samples:
            sol = solve_ivp(
                self._time_derivative,
                [0, self.time_interval],
                sample,
                t_eval=time_points,
                method='RK45'
            )
            # Unresolved attribute reference 'y' for class
            sample_trajectories.append(sol.y)

        for i, trajectory in enumerate(sample_trajectories):
            plt.plot(time_points, trajectory.T, label=f'Sample {i + 1}')

        plt.xlabel('Time')
        plt.ylabel('Sample Value')
        plt.title('Sample Evolution Over Time')
        plt.legend()
        plt.show()


class SentenceEmbeddingDensity:
    def __init__(self, sentence_encoder, mean_embedding):
        self.sentence_encoder = sentence_encoder
        self.mean_embedding = torch.tensor(mean_embedding, dtype=torch.double)

    def encode(self, sentences):
        return self.sentence_encoder.encode(sentences, convert_to_tensor=True)

    def likelihood(self, samples):
        mean_embedding_expanded = self.mean_embedding.expand(samples.size(0), -1)

        samples_double = samples.double()

        cosine_sims = util.pytorch_cos_sim(samples_double, mean_embedding_expanded)
        prob_values = torch.nn.functional.softmax(cosine_sims, dim=0)
        likelihood_values = prob_values[:, 0].detach().numpy()
        return likelihood_values


if __name__ == "__main__":
    sentence_encoder = SentenceTransformer("bert-base-nli-mean-tokens")

    initial_sentence = "initial sentence"
    initial_mean_embedding = sentence_encoder.encode([initial_sentence])
    initial_model = SentenceEmbeddingDensity(sentence_encoder, initial_mean_embedding)

    final_sentence = "final sentence"
    final_mean_embedding = sentence_encoder.encode([final_sentence])
    final_model = SentenceEmbeddingDensity(sentence_encoder, final_mean_embedding)

    time_interval = 1.0
    time_step = 0.01

    model = TextStochasticInterpolantModel(initial_model, final_model, time_interval, time_step)
    samples = model.generate_samples(num_samples=1000)
    likelihood = model.likelihood(samples)

    # model.visualize_sample_evolution(num_samples = 5) 

    print(f"Likelihood given samples: {likelihood}")
    print(f"Samples: {samples}")
