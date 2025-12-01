# Backpropagation – Part 1

## Overview
This notebook explains the **forward and backward propagation** process of a simple neural network through both **mathematical derivation** and **spreadsheet implementation**.  
The goal is to compute how the network learns by adjusting weights to minimize the total error.

This Markdown file contains:
- Complete forward and backward pass equations  
- Gradient derivations  
- Visual diagrams  
- Excel-based loss tracking  
- Learning rate comparisons  

---

## 1. Network Architecture

| Layer | Neurons | Activation | Description |
|:------|:---------|:------------|:-------------|
| Input Layer | i₁, i₂ | — | Two input features |
| Hidden Layer | h₁, h₂ | σ (sigmoid) | Hidden activations |
| Output Layer | o₁, o₂ | σ (sigmoid) | Two outputs |
| Loss | E₁, E₂ | MSE | Mean-Squared Error per output |

### Diagram
![Backpropagation Network](BP/bp.png)

The model consists of:
- **2 input nodes (i₁, i₂)**  
- **2 hidden neurons (h₁, h₂)**  
- **2 output neurons (o₁, o₂)**  
- **8 weights (w₁–w₈)** connecting all layers.

---

## 2. Forward Propagation

### Equations
```text
h1 = w1*i1 + w2*i2
h2 = w3*i1 + w4*i2

a_h1 = σ(h1) = 1 / (1 + e^(−h1))
a_h2 = σ(h2) = 1 / (1 + e^(−h2))

o1 = w5*a_h1 + w6*a_h2
o2 = w7*a_h1 + w8*a_h2

a_o1 = σ(o1)
a_o2 = σ(o2)

E1 = ½ (t1 − a_o1)²
E2 = ½ (t2 − a_o2)²
E_total = E1 + E2

## 3. Terminologies

This section defines all symbols and notations used throughout the forward and backward propagation derivations.

| **Symbol** | **Definition** | **Layer** | **Notes** |
|:------------|:----------------|:-----------|:------------|
| **i₁, i₂** | Input feature values | Input Layer | Given input data for the network |
| **w₁–w₄** | Weights connecting input → hidden layer | Between Input–Hidden | Responsible for linear transformation of inputs |
| **h₁, h₂** | Weighted sum of inputs (pre-activation) | Hidden Layer | \( h₁ = w₁i₁ + w₂i₂ \), \( h₂ = w₃i₁ + w₄i₂ \) |
| **aₕ₁, aₕ₂** | Activated hidden layer outputs | Hidden Layer | \( aₕ = σ(h) = \frac{1}{1 + e^{-h}} \) |
| **w₅–w₈** | Weights connecting hidden → output layer | Between Hidden–Output | Used to compute output layer weighted sums |
| **o₁, o₂** | Weighted sum at output neurons (pre-activation) | Output Layer | \( o₁ = w₅aₕ₁ + w₆aₕ₂ \), \( o₂ = w₇aₕ₁ + w₈aₕ₂ \) |
| **aₒ₁, aₒ₂** | Activated output neuron values | Output Layer | \( aₒ = σ(o) = \frac{1}{1 + e^{-o}} \) |
| **t₁, t₂** | Target outputs | Output Layer | Ground truth labels for comparison |
| **E₁, E₂** | Individual output errors | Loss Computation | \( E₁ = \frac{1}{2}(t₁ - aₒ₁)^2 \), \( E₂ = \frac{1}{2}(t₂ - aₒ₂)^2 \) |
| **E_total** | Total error (sum of all output errors) | Loss Function | \( E_{total} = E₁ + E₂ \) |
| **η (eta)** | Learning rate | Training Parameter | Controls update magnitude for each iteration |
| **σ(x)** | Sigmoid activation function | Activation Function | \( σ(x) = \frac{1}{1 + e^{-x}} \) |
| **∂E/∂w** | Partial derivative of loss w.r.t. weight | Gradient | Used to compute weight updates |
| **Δw** | Weight update | Optimization Step | \( Δw = -η \frac{∂E}{∂w} \) |

---

### Notes
- **Hidden Layer:** transforms inputs into intermediate representations.  
- **Output Layer:** produces final predictions through a sigmoid activation.  
- **Loss Function:** quantifies the difference between prediction and target.  
- **Gradients:** measure how much each weight influences the error.  

---

## 3. Terminologies

This section defines all symbols used throughout forward and backward propagation.

| **Symbol** | **Meaning** |
|:------------|:------------|
| **i₁, i₂** | Input features |
| **h₁, h₂** | Weighted hidden neuron sums |
| **aₕ₁, aₕ₂** | Activated hidden outputs |
| **o₁, o₂** | Weighted output neuron sums |
| **aₒ₁, aₒ₂** | Activated output values |
| **t₁, t₂** | Target (expected) values |
| **E₁, E₂** | Individual output errors |
| **E<sub>total</sub>** | Total loss = (E₁ + E₂) |

---

### Mathematical Formulation

For the total error to be minimized, we must compute partial derivatives of  
$E_{total}$ with respect to all weights $w₁$ – $w₈$:

$$
\frac{\partial E_{total}}{\partial w_i}, \quad i = 1,2,\dots,8
$$

---

## 4. Forward Pass Visualization

### Initial Weight Setup and Flow

1. Inputs ($i₁$, $i₂$) are multiplied by weights $w₁$–$w₄$ to produce $h₁$, $h₂$.  
2. Apply the sigmoid activation function:
   $$
   a_{h} = \sigma(h) = \frac{1}{1 + e^{-h}}
   $$
3. Outputs from hidden neurons ($a_{h₁}$, $a_{h₂}$) connect to output weights $w₅$–$w₈$.  
4. Compute output sums $o₁$, $o₂$ and apply activation:
   $$
   a_o = \sigma(o) = \frac{1}{1 + e^{-o}}
   $$
5. Compute losses for each output neuron using Mean Squared Error:
   $$
   E_k = \frac{1}{2}(t_k - a_{ok})^2
   $$
6. Total loss:
   $$
   E_{total} = E_1 + E_2
   $$

---

### Diagram – Forward Propagation

![Forward Flow](BP/BP4.png)

---

### Summary of Forward Computation

| **Computation** | **Equation** |
|:-----------------|:-------------|
| Hidden sum | $h_j = \sum_i w_{ij} i_i$ |
| Hidden activation | $a_{h_j} = \sigma(h_j)$ |
| Output sum | $o_k = \sum_j w_{jk} a_{h_j}$ |
| Output activation | $a_{o_k} = \sigma(o_k)$ |
| Loss | $E_{total} = \tfrac{1}{2} \sum_k (t_k - a_{o_k})^2$ |

---

# Backpropagation — Detailed Explanation (Part 2)

## 5. Backpropagation ( Derivations )

Backpropagation applies the **chain rule of calculus** to compute gradients of the total error  
with respect to each network weight.

---

### Step 1 – Derivatives for Output Layer

$$
\begin{aligned}
\frac{\partial E_{total}}{\partial w_5} &= (a_{o1} - t_1)\, a_{o1}(1 - a_{o1})\, a_{h1} \\
\frac{\partial E_{total}}{\partial w_6} &= (a_{o1} - t_1)\, a_{o1}(1 - a_{o1})\, a_{h2} \\
\frac{\partial E_{total}}{\partial w_7} &= (a_{o2} - t_2)\, a_{o2}(1 - a_{o2})\, a_{h1} \\
\frac{\partial E_{total}}{\partial w_8} &= (a_{o2} - t_2)\, a_{o2}(1 - a_{o2})\, a_{h2}
\end{aligned}
$$

---

### Step 2 – Hidden Layer Gradients

$$
\begin{aligned}
\frac{\partial E_{total}}{\partial a_{h1}} &= (a_{o1} - t_1)\, a_{o1}(1 - a_{o1})\, w_5 \;+\; (a_{o2} - t_2)\, a_{o2}(1 - a_{o2})\, w_7 \\
\frac{\partial E_{total}}{\partial a_{h2}} &= (a_{o1} - t_1)\, a_{o1}(1 - a_{o1})\, w_6 \;+\; (a_{o2} - t_2)\, a_{o2}(1 - a_{o2})\, w_8
\end{aligned}
$$

---

### Step 3 – Derivatives with Respect to Input Weights

$$
\begin{aligned}
\frac{\partial E_{total}}{\partial w_1} &= \frac{\partial E_{total}}{\partial a_{h1}}\, a_{h1}(1 - a_{h1})\, i_1 \\
\frac{\partial E_{total}}{\partial w_2} &= \frac{\partial E_{total}}{\partial a_{h1}}\, a_{h1}(1 - a_{h1})\, i_2 \\
\frac{\partial E_{total}}{\partial w_3} &= \frac{\partial E_{total}}{\partial a_{h2}}\, a_{h2}(1 - a_{h2})\, i_1 \\
\frac{\partial E_{total}}{\partial w_4} &= \frac{\partial E_{total}}{\partial a_{h2}}\, a_{h2}(1 - a_{h2})\, i_2
\end{aligned}
$$

---

### Step 4 – Consolidated Chain Rule Summary

#### Forward Pass
$$
\begin{aligned}
h_j &= \sum_i (w_{ij}\, i_i) \\
a_{h_j} &= \sigma(h_j) \\
o_k &= \sum_j (w_{jk}\, a_{h_j}) \\
a_{o_k} &= \sigma(o_k) \\
E_{total} &= \frac{1}{2} \sum_k (t_k - a_{o_k})^2
\end{aligned}
$$

#### Backward Pass
$$
\begin{aligned}
\delta_k &= (a_{o_k} - t_k)\, a_{o_k}(1 - a_{o_k}) \\
\delta_j &= \Big(\sum_k \delta_k\, w_{jk}\Big)\, a_{h_j}(1 - a_{h_j}) \\
\Delta w_{jk} &= -\eta\, \frac{\partial E_{total}}{\partial w_{jk}} = \eta\, \delta_k\, a_{h_j} \\
\Delta w_{ij} &= -\eta\, \frac{\partial E_{total}}{\partial w_{ij}} = \eta\, \delta_j\, i_i
\end{aligned}
$$

---

### Weight Update Equation

Each weight is updated according to:

$$
w^{new} = w^{old} - \eta \, \frac{\partial E_{total}}{\partial w}
$$

where $\eta$ is the **learning rate**.

---

### Visualization of Gradient Flow

![Gradient Flow Diagram](BP/BP5.png)

---

### Summary of Steps
1. Compute **forward pass** (activations and outputs).  
2. Evaluate **total loss** $E_{total}$.  
3. Compute **gradients** for output weights ($w₅$–$w₈$).  
4. Propagate gradients back to hidden weights ($w₁$–$w₄$).  
5. Update all weights using the learning rate η.  

---

# Backpropagation — Detailed Explanation (Part 3)

## 6. Weight Update Rule

Every iteration of training updates the network’s weights to minimize the total loss.

### General Update Equation
$$
w_{new} = w_{old} - \eta \, \frac{\partial E_{total}}{\partial w}
$$

where:

| **Symbol** | **Meaning** |
|:------------|:------------|
| $w$ | Weight parameter being updated |
| $\eta$ | Learning rate controlling the update step |
| $\frac{\partial E_{total}}{\partial w}$ | Gradient of loss w.r.t. the weight |

### Concept
- If $\frac{\partial E_{total}}{\partial w} > 0$, decrease the weight.  
- If $\frac{\partial E_{total}}{\partial w} < 0$, increase the weight.  
- The magnitude of the update depends on **η (learning rate)**.

### Visualization
![Weight Update Process](bp.png)

---

## 7. Spreadsheet Implementation

The accompanying Excel sheet **`bp.xlsx`** demonstrates how each gradient and weight update evolves during training.

### Excel Contents
| **Sheet Component** | **Description** |
|:---------------------|:----------------|
| Weight columns ($w_1$–$w_8$) | Iterative weight values across epochs |
| Activation columns ($a_{h1}$, $a_{h2}$, $a_{o1}$, $a_{o2}$) | Hidden & output activations |
| Gradient columns ($\partial E / \partial w_i$) | Computed partial derivatives |
| Error metrics ($E_1$, $E_2$, $E_{total}$) | Loss values per iteration |
| Learning Rate (η) | Adjustable parameter in the header |
| **Loss Graph** | Visual convergence of $E_{total}$ over time |

### Screenshot Example
![Loss Graph](bp.png)

---

## 8. Learning Rate Experiments

Different learning rates (η) were tested to analyze convergence speed and stability.

| **Learning Rate (η)** | **Behavior / Observation** |
|:-----------------------|:---------------------------|
| 0.1 | Stable but slow convergence |
| 0.2 | Faster convergence, smooth decay |
| 0.5 | Optimal speed–stability trade-off |
| 0.8 | Slight oscillation, partial overshoot |
| 1.0 | Unstable convergence |
| 2.0 | Divergent (error increases) |

### Visual Comparison
Each loss curve (from Excel) demonstrates how the choice of η affects $E_{total}$:

$$
E_{total}^{(t)} = \frac{1}{2}\sum_k \big(t_k - a_{o_k}^{(t)}\big)^2
$$

Lower η → slower convergence  Higher η → instability.

---

## 9. Loss Visualization and Interpretation

**Observation from Excel:**
- The loss curve gradually decreases across epochs, confirming correct implementation of gradient descent.  
- Occasional oscillations appear at higher learning rates.

**Graphical Trend:**
![Loss Curve Across Iterations](bp.png)

---

## 10. Key Insights

1. **Sigmoid activation** introduces non-linearity but may cause vanishing gradients.  
2. Proper **gradient flow** is essential for stable weight updates.  
3. **Learning rate tuning** governs convergence quality.  
4. **Backpropagation** mathematically ensures each parameter moves toward minimizing $E_{total}$.  
5. **Visualization in Excel** clarifies the relationship between parameter change and error reduction.

---