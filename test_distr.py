import numpy as np
import matplotlib.pyplot as plt

def main():
    mu, sigma = 3., 0.7  # mean and standard deviation
    s = 1 - np.random.lognormal(mu, sigma, 1000)

    plt.hist(s, 100, density=True, align='mid')

    plt.axis('tight')
    alpha = 0.01
    average = np.mean(s)
    cvar = np.mean(s[s < np.quantile(s, alpha)])
    var = np.quantile(s, alpha)

    plt.axvline(x=float(average), color='r', linestyle='--', label='mean')
    plt.axvline(x=float(cvar), color='b', linestyle='--', label=f'{int(alpha*100)}%-CVaR')
    plt.axvline(x=float(var), color='g', linestyle='--', label=f'{int(alpha*100)}%-VaR')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()