import subprocess
import sys

# Function to install missing packages
def install_package(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install required packages
required_packages = ["matplotlib", "streamlit", "numpy", "scipy"]
for package in required_packages:
    install_package(package)

# Import the installed modules
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Function to calculate the Black-Scholes call and put prices
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option_type. Use 'call' or 'put'.")
    return price

# Streamlit app
st.title("Black-Scholes Model: Interactive Heatmaps")
st.markdown("Visualize Call and Put option prices using the Black-Scholes model with interactive heatmaps.")

# Inputs
K = st.number_input("Strike Price (K)", min_value=1.0, value=100.0, step=1.0)
T = st.number_input("Time to Maturity (T, in years)", min_value=0.01, value=1.0, step=0.01)
r = st.number_input("Risk-Free Rate (r, in decimal)", min_value=0.0, value=0.05, step=0.01)
spot_prices = st.text_input("Spot Prices (comma-separated)", value="50,100,150")
volatilities = st.text_input("Volatilities (comma-separated)", value="0.1,0.2,0.3")

# Convert inputs
spot_prices = np.array([float(s) for s in spot_prices.split(",") if s.strip()])
volatilities = np.array([float(v) for v in volatilities.split(",") if v.strip()])

# Create heatmaps for call and put prices
call_prices = np.array([[black_scholes(S, K, T, r, sigma, "call") for sigma in volatilities] for S in spot_prices])
put_prices = np.array([[black_scholes(S, K, T, r, sigma, "put") for sigma in volatilities] for S in spot_prices])

# Plot heatmaps
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

c1 = ax1.imshow(call_prices, extent=[volatilities.min(), volatilities.max(), spot_prices.min(), spot_prices.max()],
                origin='lower', aspect='auto', cmap='viridis')
ax1.set_title("Call Option Prices Heatmap")
ax1.set_xlabel("Volatility (σ)")
ax1.set_ylabel("Spot Price (S)")
fig.colorbar(c1, ax=ax1)

c2 = ax2.imshow(put_prices, extent=[volatilities.min(), volatilities.max(), spot_prices.min(), spot_prices.max()],
                origin='lower', aspect='auto', cmap='viridis')
ax2.set_title("Put Option Prices Heatmap")
ax2.set_xlabel("Volatility (σ)")
ax2.set_ylabel("Spot Price (S)")
fig.colorbar(c2, ax=ax2)

# Display the plots in Streamlit
st.pyplot(fig)

