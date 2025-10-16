# ⚛️ Classical vs Quantum SVM (PennyLane)

An interactive **Streamlit web application** comparing the performance of **Classical SVM (RBF Kernel)** and **Quantum SVM (PennyLane-based Quantum Kernel)** across various datasets, with adjustable parameters, visualizations, and evaluation metrics.

---

## 🚀 Features

### 🧠 Model Comparison
- **Classical SVM (RBF Kernel):**  
  Implements a standard radial basis function kernel:
  \[
  K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)
  \]

- **Quantum SVM:**  
  Uses **PennyLane’s default.qubit simulator** to encode data into quantum states using parameterized rotations:
  \[
  K(x, y) = |\langle \psi(x) | \psi(y) \rangle|^2
  \]

### 📊 Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Processing Time  

### 🧩 Interactive Dataset Options
- Upload your own CSV file  
- Preloaded sample datasets:
  - Iris
  - Glass
  - Social Network Ads

### 🧮 Preprocessing & Configurations
- Automatic label encoding and one-hot encoding  
- StandardScaler normalization  
- Optional PCA (Principal Component Analysis)  
- Adjustable test split ratio and quantum training sample limits

### 📈 Visualization Tools
- Confusion matrices for both models  
- Decision boundaries (for 2D data)  
- Metric comparison bar charts  
- Metric variation across multiple dataset splits  
- Cross-dataset summary with trend plots and bar comparisons  

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| Frontend/UI | **Streamlit** |
| Classical ML | **Scikit-learn (SVM, PCA, Metrics)** |
| Quantum ML | **PennyLane (default.qubit backend)** |
| Visualization | **Matplotlib, Seaborn** |
| Data Handling | **Pandas, NumPy** |

---

## 📂 Project Structure

```
📦 classical-vs-quantum-svm
│
├── app.py                           # Main Streamlit application
├── datasets/                        # Dataset storage
│   ├── iris.csv
│   ├── glass.csv
│   ├── social_network_ads.csv
│   └── dataset_summary_results.csv  # Optional summary data
│
├── README.md                        # Project documentation
```

---

## 🧮 Adjustable Parameters (Sidebar Controls)

| Parameter | Description |
|------------|--------------|
| PCA Components | Reduce feature dimensions for visualization and quantum kernel stability |
| Test Size | Fraction of dataset used for testing |
| Max Quantum Samples | Limits dataset size for quantum circuit simulation |
| Quantum Limit Factor | Simulates hardware constraints by reducing training data |
| Dataset Choice | Select from preloaded datasets or upload your own CSV |

---

## 🧩 Outputs & Visualizations

| Section | Description |
|----------|--------------|
| **Metric Dashboard** | Displays accuracy, precision, recall, F1, and time for both models |
| **Confusion Matrices** | Heatmaps for both Classical and Quantum predictions |
| **Decision Boundary** | For 2D data, shows model separation visually |
| **Metric Comparison Chart** | Side-by-side bar chart comparison |
| **Metric Variation Across Splits** | Shows metric fluctuations across 10 sample splits |
| **Cross-Dataset Summary** | Aggregates results across datasets for deeper insight |
| **Trend Plots & Bar Charts** | Compare Classical vs Quantum across dataset sizes |

---

## 📚 Conceptual Overview

This project demonstrates:
- The practical **differences between classical kernel methods and quantum-enhanced kernels**.  
- How **quantum circuits can be used to construct data-dependent kernels** for classification tasks.  
- The trade-offs between **accuracy and computational cost** under varying dataset sizes and quantum limitations.

It serves as a foundation for **Quantum-Assisted Machine Learning (QAML)** exploration and benchmarking.

---

## 🧠 Research Relevance

This project replicates and extends experiments inspired by the IEEE paper:  
**“Comparative Analysis of a Quantum SVM With an Optimized Kernel Versus Classical SVMs.”**

It provides a real-time, visual, and interactive environment to analyze both models’ behavior under identical conditions.

---

## 📜 License

This project is released under the **MIT License**.  
You are free to use, modify, and distribute it with attribution.

---

## 👨‍💻 Author

**PlatinumManX**  
🎓 Engineering Student | 💻 Technical Game Dev Enthusiast | ⚛️ Quantum ML Explorer  
📫 Connect: [GitHub Profile](https://github.com/PlatinumManX)
