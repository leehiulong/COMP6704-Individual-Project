# Comparative Analysis of First-Order Methods for Sparse Signal Recovery in Compressed Sensing

This repository contains the implementation and experimental evaluation of first-order optimization methods for compressed sensing, as presented in the paper "Comparative Analysis of First-Order Methods for Sparse Signal Recovery in Compressed Sensing".

## 📋 Project Overview

This project implements and compares four optimization algorithms for solving the LASSO problem in compressed sensing:
- **ISTA** (Iterative Shrinkage-Thresholding Algorithm)
- **FISTA** (Fast Iterative Shrinkage-Thresholding Algorithm) 
- **ADMM with Woodbury Identity**
- **ADMM with Conjugate Gradient**

## 🚀 Installation

1. Clone this repository:
```bash
git clone https://github.com/leehiulong/COMP6704-Individual-Project.git
cd COMP6704-Individual-Project
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Usage
Run the main experiments:
```python
python experiments.py
```
The script will:
- Generate synthetic compressed sensing problems
- Run all optimization algorithms
- Generate comprehensive visualizations
- Save results to the results/ directory

## 📊 Results
The experiments evaluate algorithms across 8 scenarios with varying:
- Compression ratios (m/n = 0.25, 0.7)
- Sparsity levels (k/n = 0.05, 0.1)
- Noise conditions (σ = 0.01, 0.1)
Key metrics evaluated:
- Reconstruction accuracy
- Computational efficiency
- Convergence behavior
- Algorithm stability

## 📁 Project Structure
```text
COMP6704-Individual-Project/
├── experiments.py           # Experimental codes in Python
├── report.pdf              # Report PDF
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── results/                # Generated results and plots
```

## 📈 Key Findings
- FISTA provides the best speed-accuracy tradeoff
- ISTA offers the highest stability for critical applications
- ADMM with CG balances flexibility and performance
- All algorithms show fundamental limits under low compression, high sparsity, high noise conditions

## 📚 Citation
If you use this code in your research, please cite:

```bibtex
@article{lee2025compressed,
  title={Comparative Analysis of First-Order Methods for Sparse Signal Recovery in Compressed Sensing},
  author={Hiu Long Lee},
  journal={COMP6704 Individual Project Report},
  year={2025}
}
```

## 🤝 Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for suggestions and bug reports.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
