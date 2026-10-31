# Overfitting and Regularization

**The Math Behind Models That Generalize**

A model that is perfect on its training data can be useless on new data. This project explores why that happens and how to detect and prevent it, using the mathematics behind machine learning: linear algebra, calculus and probability. Every method is derived step by step, implemented from scratch with NumPy and verified against scikit-learn.

Final project for the **Math Concepts for Developers** course.

## Main results

- **Overfitting:** on 21 noisy points from $\sin(2\pi x)$, a polynomial of degree 15 passes close to every training point, but its test MSE is 1.82 and its largest weight is about $5.8 \times 10^6$.
- **Bias-variance:** a simulation with 200 datasets confirms the decomposition $\text{Bias}^2 + \text{Variance} + \sigma^2$ and explains the U-shaped test error.
- **Regularization:** Ridge regression keeps the degree-15 polynomial, but shrinks its weights. With $\lambda$ chosen by cross-validation, its test MSE drops to 0.026.
- **Model selection:** 5-fold cross-validation chooses degree 3 using only the training data, and makes the same choice on 162 of 200 random splits, while a single 9-point test set would have chosen degree 7 by luck.
- **Real data:** on the diabetes dataset (442 patients), regularization makes no difference for a linear model with 10 features, but prevents overfitting with 64 quadratic features. Lasso identifies body mass index, blood pressure and a triglyceride-related blood measurement as the most important features.

## Contents of the notebook

| Section | Topic |
|---|---|
| 1. Introduction | problem formulation, objectives, scope and assumptions |
| 2–3. Setup and Data | synthetic data from a noisy sine, train/test split |
| 4. Linear Regression from Scratch | normal equation, maximum likelihood and the MSE loss |
| 5. Polynomial Regression | the Vandermonde design matrix |
| 6. Overfitting and Underfitting | train and test error, exploding weights |
| 7. The Bias-Variance Tradeoff | derivation and simulation |
| 8. Regularization: Ridge and Lasso | closed-form Ridge solution, invertibility proof, coefficient paths, soft thresholding, L1/L2 geometry |
| 9. Model Selection with Cross-Validation | k-fold cross-validation for the degree and $\lambda$ |
| 10. Real-World Application | OLS, Ridge and Lasso on the diabetes dataset |
| 11–12. Discussion and Conclusion | limitations, repeated-split experiment, next steps |
| 13. References | all cited sources |

**Implemented from scratch:** train/test split, linear regression (normal equation), polynomial features, Ridge regression and k-fold cross-validation.
**From scikit-learn:** Lasso (no closed-form solution), quadratic features for the diabetes data, the diabetes dataset, and the reference implementations used for the checks.

## How to run

1. Install Python 3.10 or newer.
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Open `main.ipynb` in Jupyter or VS Code and select **Restart & Run All**. The whole notebook runs in about 15 seconds.

The diabetes dataset is included in scikit-learn, so no download is needed. All random operations use fixed seeds, so the results are reproducible. The notebook was tested with Python 3.14, NumPy 2.4, Matplotlib 3.10 and scikit-learn 1.8.

## Repository structure

```
main.ipynb         the project notebook
requirements.txt   Python dependencies
README.md          this file
LICENSE            MIT license
```

## Main references

- C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006.
- M. P. Deisenroth, A. A. Faisal and C. S. Ong, *Mathematics for Machine Learning*, Cambridge University Press, 2020.
- T. Hastie, R. Tibshirani and J. Friedman, *The Elements of Statistical Learning*, 2nd ed., Springer, 2009.
- A. E. Hoerl and R. W. Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal Problems," *Technometrics*, 1970.
- R. Tibshirani, "Regression Shrinkage and Selection via the Lasso," *Journal of the Royal Statistical Society: Series B*, 1996.

The full list of references is in Section 13 of the notebook.

## Author

Eleonora Kuteva
