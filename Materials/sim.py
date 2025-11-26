import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_call_risk_neutral(S0, K, T, r, sigma, N_sims):
    np.random.seed(42)
    payoffs = []
    prices = []
    for i in range(1, N_sims + 1):
        Z = np.random.randn()
        S_T = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
        payoff = max(S_T - K, 0)
        payoffs.append(payoff)
        # Track running estimate
        expected_payoff = np.mean(payoffs)
        option_price = np.exp(-r * T) * expected_payoff
        prices.append(option_price)
    return prices

# Parameters
S0 = 100
K = 100
T = 1
r = 0.05
sigma = 0.2
N_sims = 1000

prices = monte_carlo_call_risk_neutral(S0, K, T, r, sigma, N_sims)

plt.figure(figsize=(10,6))
plt.plot(prices, label='Monte Carlo Estimate')
plt.axhline(y=prices[-1], color='r', linestyle='--', label='Final Price')
plt.xlabel('Number of Simulations')
plt.ylabel('Call Option Price')
plt.title('Convergence of Monte Carlo Option Price')
plt.legend()
plt.show()
