Overview

The Polynomial Data Science Toolkit is a sophisticated Python implementation that leverages doubly linked lists to represent and manipulate sparse polynomials for advanced data science applications. This single-file repository provides a comprehensive framework for polynomial operations, regression, feature engineering, and visualization, with a focus on memory-efficient handling of sparse, high-degree polynomials. It is designed for data scientists, researchers, and engineers seeking to model complex relationships in datasets, such as time-series forecasting, curve fitting, or statistical analysis.

Key highlights include:





Doubly Linked Lists: Efficiently store and manipulate sparse polynomials with O(n) insertion/deletion, ideal for memory-constrained environments.



Polynomial Regression: Fits data with customizable degrees and L1/L2 regularization, optimized for sparse polynomial representations.



Feature Engineering: Generates polynomial and interaction features for machine learning pipelines.



Visualization: Produces insightful plots of polynomial functions, regression fits, and training loss curves using Matplotlib.

DEMO:

=== Polynomial Operations Demo ===
p1 = 3.00x^2 + -2.00x + 5.00
p2 = 1.00x^3 + 4.00x

p1 + p2 = 1.00x^3 + 3.00x^2 + 2.00x + 5.00

p1 * p2 = 3.00x^5 + -2.00x^4 + 17.00x^3 + -8.00x^2 + 20.00x

d/dx(p1) = 6.00x + -2.00

∫p1 dx = 1.00x^3 + -1.00x^2 + 5.00x + C

![image](https://github.com/user-attachments/assets/39d5d49d-939c-4b64-89dd-d232c1c0fd65)

=== Polynomial Regression Demo ===
Epoch 0: MSE = 125.4614
Epoch 100: MSE = 3.3749
Epoch 200: MSE = 1.4656
Epoch 300: MSE = 1.3530
Epoch 400: MSE = 1.2989
Epoch 500: MSE = 1.2511
Epoch 600: MSE = 1.2077
Epoch 700: MSE = 1.1682
Epoch 800: MSE = 1.1323
Epoch 900: MSE = 1.0997
Epoch 1000: MSE = 1.0700
Epoch 1100: MSE = 1.0431
Epoch 1200: MSE = 1.0185
Epoch 1300: MSE = 0.9962
Epoch 1400: MSE = 0.9759
Epoch 1500: MSE = 0.9575
Epoch 1600: MSE = 0.9407
Epoch 1700: MSE = 0.9255
Epoch 1800: MSE = 0.9116
Epoch 1900: MSE = 0.8990
Epoch 2000: MSE = 0.8875
Epoch 2100: MSE = 0.8771
Epoch 2200: MSE = 0.8676
Epoch 2300: MSE = 0.8590
Epoch 2400: MSE = 0.8511
Epoch 2500: MSE = 0.8440
Epoch 2600: MSE = 0.8375
Epoch 2700: MSE = 0.8316
Epoch 2800: MSE = 0.8262
Epoch 2900: MSE = 0.8214
Epoch 3000: MSE = 0.8169
Epoch 3100: MSE = 0.8129
Epoch 3200: MSE = 0.8092
Epoch 3300: MSE = 0.8059
Epoch 3400: MSE = 0.8029
Epoch 3500: MSE = 0.8001
Epoch 3600: MSE = 0.7976
Epoch 3700: MSE = 0.7953
Epoch 3800: MSE = 0.7932
Epoch 3900: MSE = 0.7913
Epoch 4000: MSE = 0.7896
Epoch 4100: MSE = 0.7881
Epoch 4200: MSE = 0.7866
Epoch 4300: MSE = 0.7853
Epoch 4400: MSE = 0.7842
Epoch 4500: MSE = 0.7831
Epoch 4600: MSE = 0.7821
Epoch 4700: MSE = 0.7812
Epoch 4800: MSE = 0.7804
Epoch 4900: MSE = 0.7797
Epoch 4999: MSE = 0.7791



![image](https://github.com/user-attachments/assets/64193b65-0f10-41e4-9407-3a43a8f0d020)


![image](https://github.com/user-attachments/assets/f2d26459-6dd1-430a-8159-baa19c6540d1)

=== Feature Engineering Demo ===
Original features:
[[1 2]
 [3 4]
 [5 6]]

Polynomial features:
[[ 1  1  2  4  2]
 [ 3  9  4 16 12]
 [ 5 25  6 36 30]]

 


