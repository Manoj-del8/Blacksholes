import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Function for Black-Scholes call and put option prices
def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Calculate the Black-Scholes price for a call or put option.

    Parameters:
    - S: Spot price
    - K: Strike price
    - T: Time to maturity (in years)
    - r: Risk-free interest rate
    - sigma: Volatility (standard deviation of returns)
    - option_type: 'call' or 'put'

    Returns:
    - Option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")
    return price

# Streamlit App Title
st.title("Black-Scholes Model: Heatmap with Sensitivity Analysis")
st.markdown("Analyze **Call** and **Put** option prices with interactive heatmaps and sensitivity analysis.")

# Sidebar Inputs
st.sidebar.header("Model Parameters")
K = st.sidebar.number_input("Strike Price (K)", min_value=1.0, value=100.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity (T, in years)", min_value=0.01, value=1.0, step=0.01)
r = st.sidebar.number_input("Risk-Free Rate (r, in decimal)", min_value=0.0, value=0.05, step=0.01)

# Spot price and volatility inputs for sensitivity analysis
st.sidebar.header("Sensitivity Analysis Inputs")
sensitivity_spot = st.sidebar.number_input("Spot Price for Sensitivity (S)", min_value=1.0, value=100.0, step=1.0)
sensitivity_vol = st.sidebar.number_input("Volatility for Sensitivity (σ)", min_value=0.01, max_value=1.0, value=0.2, step=0.01)

# Heatmap range inputs
st.sidebar.header("Heatmap Range")
spot_min = st.sidebar.number_input("Minimum Spot Price", min_value=1.0, value=50.0, step=1.0)
spot_max = st.sidebar.number_input("Maximum Spot Price", min_value=1.0, value=150.0, step=1.0)
vol_min = st.sidebar.number_input("Minimum Volatility (σ)", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
vol_max = st.sidebar.number_input("Maximum Volatility (σ)", min_value=0.01, max_value=1.0, value=0.5, step=0.01)

# Generate Spot Price and Volatility Ranges
spot_prices = np.linspace(spot_min, spot_max, 100)
volatilities = np.linspace(vol_min, vol_max, 100)

# Create 2D Grids of Call and Put Option Prices
call_prices = np.zeros((len(volatilities), len(spot_prices)))
put_prices = np.zeros((len(volatilities), len(spot_prices)))

for i, sigma in enumerate(volatilities):
    for j, spot in enumerate(spot_prices):
        call_prices[i, j] = black_scholes(spot, K, T, r, sigma, option_type="call")
        put_prices[i, j] = black_scholes(spot, K, T, r, sigma, option_type="put")

# Sensitivity Analysis Prices
call_price_sensitivity = black_scholes(sensitivity_spot, K, T, r, sensitivity_vol, option_type="call")
put_price_sensitivity = black_scholes(sensitivity_spot, K, T, r, sensitivity_vol, option_type="put")

# Display Sensitivity Analysis Results
st.subheader("Sensitivity Analysis")
st.write(f"**Call Option Price (S={sensitivity_spot}, σ={sensitivity_vol}):** {call_price_sensitivity:.2f}")
st.write(f"**Put Option Price (S={sensitivity_spot}, σ={sensitivity_vol}):** {put_price_sensitivity:.2f}")

# Call Option Heatmap
st.subheader("Call Option Price Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
c1 = ax.imshow(call_prices, extent=[spot_min, spot_max, vol_min, vol_max],
               origin='lower', aspect='auto', cmap='viridis')
ax.set_title("Call Option Price", fontsize=16)
ax.set_xlabel("Spot Price (S)", fontsize=12)
ax.set_ylabel("Volatility (σ)", fontsize=12)
ax.scatter(sensitivity_spot, sensitivity_vol, color='red', label=f"Sensitivity: (S={sensitivity_spot}, σ={sensitivity_vol})", s=100)
ax.legend()
fig.colorbar(c1, ax=ax, label="Call Option Price")
st.pyplot(fig)

# Put Option Heatmap
st.subheader("Put Option Price Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
c2 = ax.imshow(put_prices, extent=[spot_min, spot_max, vol_min, vol_max],
               origin='lower', aspect='auto', cmap='viridis')
ax.set_title("Put Option Price", fontsize=16)
ax.set_xlabel("Spot Price (S)", fontsize=12)
ax.set_ylabel("Volatility (σ)", fontsize=12)
ax.scatter(sensitivity_spot, sensitivity_vol, color='red', label=f"Sensitivity: (S={sensitivity_spot}, σ={sensitivity_vol})", s=100)
ax.legend()
fig.colorbar(c2, ax=ax, label="Put Option Price")
st.pyplot(fig)
