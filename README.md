
# 🛰️ Orbital-ML: Hybrid Physics & Neural Network Trajectory Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A computational physics experiment comparing classical Newtonian mechanics against a Feedforward Neural Network (FNN) to predict 2D satellite trajectories around Earth. This project demonstrates that while machine learning can memorize short-term physics, it fundamentally fails at long-term orbital propagation due to an inability to conserve Hamiltonian energy.

*(Insert a GIF or image of the Colab animation here showing the Cyan Physics dot vs the Red AI dot diverging)*

---

## 🎯 The Purpose

**Can an AI learn Newtonian gravity just by looking at data?** 

In recent years, Machine Learning has been applied to fluid dynamics, climate modeling, and molecular dynamics. This project investigates whether a standard neural network can replace a physics-based differential equation solver for orbital mechanics. 

By simulating a satellite in Low Earth Orbit (LEO) and training an AI to predict its next position, we expose a critical flaw in data-driven scientific models: **Standard ML models minimize Euclidean error (MSE), not physical laws.** Over long time horizons, this results in catastrophic divergence.

---

## 🧪 What We Made

We built a complete end-to-end pipeline consisting of three core modules:

1. **A High-Fidelity Physics Engine:** A numerically stable satellite simulator using the Velocity Verlet integration algorithm.
2. **An ML Forecasting Pipeline:** A data-preprocessing and sliding-window dataset generator that trains a Neural Network to act as an orbital propagator.
3. **A Real-Time Visual Analyzer:** An animated visualization (optimized for Google Colab) that compares the ground-truth physics against the AI's autoregressive predictions in real-time.

---

## ⚙️ How We Made It

### 1. The Physics Simulation (Ground Truth)
We simulated a test-particle satellite at 400km altitude (ISS-like orbit). 
* **Equation of Motion:** $\vec{a}(t) = -\frac{GM}{|\vec{r}(t)|^3}\vec{r}(t)$
* **Integrator:** We explicitly avoided the Euler method due to artificial energy drift. Instead, we used the **Velocity Verlet** algorithm. Verlet is a *symplectic integrator*, meaning it preserves the geometric structure of Hamiltonian phase space, ensuring the simulated orbit remains perfectly stable indefinitely.
* **Output:** Time-series arrays of $(x, y, v_x, v_y)$ in standard SI units.

### 2. Dataset Generation
Because we needed the AI to predict into the future recursively (autoregressively), we couldn't just predict position. We had to predict the full state vector.
* **Input Shape:** A sliding window of 10 timesteps (100 seconds) flattened into a vector of size 40: $[x, y, v_x, v_y] \times 10$.
* **Target Shape:** The immediate next state vector of size 4: $[x(t+\Delta t), y(t+\Delta t), v_x(t+\Delta t), v_y(t+\Delta t)]$.
* **Preprocessing:** Standard scaling was applied to the inputs. Targets were left unnormalized to prevent artificial scaling errors in the velocity vectors.

### 3. The Machine Learning Model
Because Newtonian gravity defines a Markovian system (the future depends only on the present), a simple Feedforward Neural Network (FNN) was chosen over a complex LSTM.
* **Architecture:** `Input(40) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(4, Linear)`
* **Loss Function:** Mean Squared Error (MSE).
* **Deployment:** To test long-term stability, the model was used *autoregressively*: it predicted step $t+1$, then fed its own prediction back in as the input for step $t+2$.

---

## 📉 Key Results & Findings

When you run the simulation, you will observe the following:

1. **Short-Term Accuracy:** The AI achieves an incredibly low single-step MSE ($< 1$ meter error). For one orbit, the AI and Physics paths overlap perfectly.
2. **Long-Term Divergence:** By the second and third orbits, the AI trajectory visibly diverges from the physics trajectory. 
3. **Orbital Precession (The Smoking Gun):** The AI doesn't spiral inward or outward randomly. Instead, it exhibits **orbital precession**—its elliptical path rotates around the Earth. 
4. **Energy Non-Conservation:** Analysis of the specific orbital energy ($E = \frac{1}{2}v^2 - \frac{GM}{r}$) shows that while the Verlet integrator conserves energy exactly, the Neural Network slowly injects artificial energy into the system because MSE loss does not enforce the laws of thermodynamics.

**Conclusion:** Pure data-driven models are fundamentally inadequate for long-horizon orbital propagation without explicit physics constraints (e.g., Physics-Informed Neural Networks).

---

## 🚀 How to Run

### Prerequisites
Ensure you have the following libraries installed:
```bash
pip install numpy matplotlib scikit-learn tensorflow
```

### Running Locally
If you are running a standard Python environment (VS Code, PyCharm), the script will open a native GUI window showing the animation:
```bash
python orbital_simulation.py
```

### Running in Google Colab (Recommended)
Colab cannot open native GUI windows. The main script is written to automatically detect the Colab environment and render the simulation as an HTML5 video directly inside the notebook cell. Just run the cell and wait ~15 seconds for the video to encode.

---


---

## 📜 License

This project is licensed under the MIT License - feel free to use this as a template for your own physics-informed ML experiments!

---
*Built with Python, Newtonian Mechanics, and a healthy skepticism of black-box AI.*
