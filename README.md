# 🚀 MLP-Classifier

An MLP Classifier implemented in C, with functions for **MSE**, **Binary Cross-Entropy**, **ReLU**, **Sigmoid**, and **weight updating through backpropagation**.

---

## 📜 Model Description

### 🧠 Input Layer
- **1 input node**: Receives the input data for 1 independent **X feature** variable.

---

### 🔒 Hidden Layer
- **2 nodes** in the hidden layer, each with its own **weight** and **bias**:
  - **Node 1**: Weight `w00`, Bias `b00`
  - **Node 2**: Weight `w01`, Bias `b01`

Each hidden node calculates its output using the **ReLU** activation function:

\[
output_i = ReLU(x_i * w + b)
\]

where `x_i` is the input feature, `w` is the weight, and `b` is the bias.

---

### 💡 Output Layer
The outputs from the two hidden nodes (`neuron_00` and `neuron_01`) are combined to form the input to the output layer:

\[
combined value = (neuron_00 * w10) + (neuron_01 * w11) + b1
\]

Where:
- `w10` and `w11` are the weights.
- `b1` is the bias.

The combined value is passed through the **Sigmoid** activation function to produce the final output (prediction **ŷ**):

\[
ŷ = Sigmoid(combined value)
\]

This output represents a probability, a value between **0** and **1**.

---

## ⚙️ Activation Functions
- **ReLU**: `ReLU(x) = x > 0 ? x : 0`
- **Sigmoid**: `Sigmoid(x) = 1 / (1 + e^(-x))`

---

## 🧪 Loss Functions
- **Mean Squared Error (MSE)**
- **Binary Cross-Entropy**

---

## 🔄 Weight Update (Backpropagation)
The model updates weights using **backpropagation** to minimize the loss function.
