import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Function for Black-Scholes call option price
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price

# Function for Black-Scholes put option price
def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

# Streamlit App Title
st.title("Black-Scholes Option Pricing Application")

# Sidebar for Inputs
st.sidebar.header("Input Parameters")
spot_price = st.sidebar.number_input("Spot Price (S)", min_value=0.01, value=100.0)
strike_price = st.sidebar.number_input("Strike Price (K)", min_value=0.01, value=100.0)
T = st.sidebar.number_input("Time to Maturity (T in years)", min_value=0.01, value=1.0)
r = st.sidebar.number_input("Risk-Free Rate (r, decimal)", min_value=0.0, value=0.05)
sigma = st.sidebar.number_input("Volatility (σ, decimal)", min_value=0.01, value=0.2)

# Sidebar for Heatmap Parameters
st.sidebar.header("Heatmap Parameters")
min_spot = st.sidebar.number_input("Minimum Spot Price", min_value=0.01, value=50.0)
max_spot = st.sidebar.number_input("Maximum Spot Price", min_value=0.01, value=150.0)
min_vol = st.sidebar.number_input("Minimum Volatility", min_value=0.01, value=0.1)
max_vol = st.sidebar.number_input("Maximum Volatility", min_value=0.01, value=0.5)
matrix_size = st.sidebar.slider("Matrix Size", min_value=5, max_value=50, value=10)
color_map = st.sidebar.selectbox("Colormap", ["viridis", "plasma", "YlOrRd", "PuBuGn", "coolwarm"])

# Calculate Call and Put Prices
call_price = black_scholes_call(spot_price, strike_price, T, r, sigma)
put_price = black_scholes_put(spot_price, strike_price, T, r, sigma)

# Display Call and Put Prices in Boxes
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-around; padding: 10px;">
        <div style="background-color: #4CAF50; padding: 20px; border-radius: 10px; text-align: center; color: white;">
            <h3>Call Option Price</h3>
            <h1>${call_price:.2f}</h1>
        </div>
        <div style="background-color: #FF5722; padding: 20px; border-radius: 10px; text-align: center; color: white;">
            <h3>Put Option Price</h3>
            <h1>${put_price:.2f}</h1>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Generate Spot Prices and Volatilities for Heatmap
spot_prices = np.linspace(min_spot, max_spot, matrix_size)
volatilities = np.linspace(min_vol, max_vol, matrix_size)

# Calculate Heatmap Data
call_prices_matrix = np.array([[black_scholes_call(S, strike_price, T, r, v) for v in volatilities] for S in spot_prices])
put_prices_matrix = np.array([[black_scholes_put(S, strike_price, T, r, v) for v in volatilities] for S in spot_prices])

# Call Option Heatmap
st.subheader("Call Option Price Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
c = ax.imshow(call_prices_matrix, extent=[min_vol, max_vol, min_spot, max_spot], origin="lower", aspect="auto", cmap=color_map)
ax.set_title("Call Option Price Sensitivity")
ax.set_xlabel("Volatility (σ)")
ax.set_ylabel("Spot Price (S)")
fig.colorbar(c, label="Call Option Price")
st.pyplot(fig)

# Put Option Heatmap
st.subheader("Put Option Price Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
c = ax.imshow(put_prices_matrix, extent=[min_vol, max_vol, min_spot, max_spot], origin="lower", aspect="auto", cmap=color_map)
ax.set_title("Put Option Price Sensitivity")
ax.set_xlabel("Volatility (σ)")
ax.set_ylabel("Spot Price (S)")
fig.colorbar(c, label="Put Option Price")
st.pyplot(fig)
