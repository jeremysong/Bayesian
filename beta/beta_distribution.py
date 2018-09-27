from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np


def beta_a_b_from_mean_kappa(mean, kappa):
    a = mean * kappa
    b = (1 - mean) * kappa
    return a, b


def beta_a_b_from_mode_kappa(mode, kappa):
    a = mode * (kappa - 2) + 1
    b = (1 - mode) * (kappa - 2) + 1
    return a, b


def beta_a_b_from_mean_sd(mean, sd):
    num = mean * (1 - mean) / sd ** 2 - 1
    a = mean * num
    b = (1 - mean) * num
    return a, b


def bernoulli_likelihood(theta, n, z):
    return theta ** z * (1 - theta) ** (n - z)


# a, b = beta_a_b_from_mean_kappa(mean=0.25, kappa=4)
# print({'a': a, 'b': b})
#
# a, b = beta_a_b_from_mode_kappa(mode=0.25, kappa=4)
# print({'a': a, 'b': b})
#
# a, b = beta_a_b_from_mean_sd(mean=0.5, sd=0.1)
# print({'a': a, 'b': b})

def beta_distribution_as_prior():
    fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)
    a, b = beta_a_b_from_mode_kappa(mode=0.75, kappa=25)

    x = np.linspace(0, 1, 1000)
    ax1.plot(x, beta.pdf(x, a, b), lw=5)
    ax1.set_title('Prior (beta)')

    ax2.plot(x, bernoulli_likelihood(x, 20, 17))
    ax2.set_title('Likelihood (Bernoulli)')

    plt.show()


def abnormal_prior():
    x = np.linspace(0, 1, 1000)
    y_raw = np.concatenate((np.repeat(1, 200), np.linspace(1, 100, 50), np.linspace(100, 1, 50), np.repeat(1, 200),
                            np.repeat(1, 200), np.linspace(1, 100, 50), np.linspace(100, 1, 50), np.repeat(1, 200)))
    prior = y_raw / np.sum(y_raw)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True)
    ax1.plot(x, prior)
    ax1.set_title('Prior')

    likelihood = [bernoulli_likelihood(theta, 27, 14) for theta in x]
    ax2.plot(x, likelihood)
    ax2.set_title('Likelihood (Bernoulli)')

    posterior_raw = prior * likelihood
    posterior = posterior_raw / sum(posterior_raw)
    ax3.plot(x, posterior)
    ax3.set_title('Posterior')

    plt.show()


if __name__ == '__main__':
    abnormal_prior()
