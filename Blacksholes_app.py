import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Function for Black-Scholes call option price
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price

# Function for Black-Scholes put option price
def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

# Streamlit App Title
st.title("Black-Scholes Model App")

# Input Fields
S = st.number_input("Spot Price (S)", min_value=0.01, value=100.0)
K = st.number_input("Strike Price (K)", min_value=0.01, value=100.0)
T = st.number_input("Time to Maturity (T in years)", min_value=0.01, value=1.0)
r = st.number_input("Risk-Free Rate (r, decimal)", min_value=0.0, value=0.05)
sigma = st.number_input("Volatility (σ, decimal)", min_value=0.01, value=0.2)

# Calculate Option Prices
call_price = black_scholes_call(S, K, T, r, sigma)
put_price = black_scholes_put(S, K, T, r, sigma)

st.write(f"Call Option Price: {call_price:.2f}")
st.write(f"Put Option Price: {put_price:.2f}")

# Heatmap
spot_prices = np.linspace(S * 0.5, S * 1.5, 50)
volatilities = np.linspace(0.01, 1.0, 50)

call_prices = np.array([[black_scholes_call(S, K, T, r, v) for v in volatilities] for S in spot_prices])
put_prices = np.array([[black_scholes_put(S, K, T, r, v) for v in volatilities] for S in spot_prices])

# Plot Call Price Heatmap
fig, ax = plt.subplots()
c = ax.imshow(call_prices, extent=[0.01, 1.0, S * 0.5, S * 1.5], origin="lower", aspect="auto", cmap="viridis")
ax.set_title("Call Option Price Heatmap")
ax.set_xlabel("Volatility (σ)")
ax.set_ylabel("Spot Price (S)")
fig.colorbar(c, label="Call Option Price")
st.pyplot(fig)

# Plot Put Price Heatmap
fig, ax = plt.subplots()
c = ax.imshow(put_prices, extent=[0.01, 1.0, S * 0.5, S * 1.5], origin="lower", aspect="auto", cmap="viridis")
ax.set_title("Put Option Price Heatmap")
ax.set_xlabel("Volatility (σ)")
ax.set_ylabel("Spot Price (S)")
fig.colorbar(c, label="Put Option Price")
st.pyplot(fig)


