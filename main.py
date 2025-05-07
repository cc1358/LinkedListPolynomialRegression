"""
Polynomial Data Science Toolkit
A comprehensive implementation of polynomial operations using linked lists
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from sklearn.metrics import mean_squared_error

class Node:
    """Doubly linked list node for polynomial terms"""
    def __init__(self, coefficient, exponent):
        self.coefficient = coefficient
        self.exponent = exponent
        self.prev = None
        self.next = None

class Polynomial:
    """Polynomial implementation using doubly linked list"""
    def __init__(self):
        self.head = None
        self.tail = None
        self._length = 0

    def insert_term(self, coefficient, exponent):
        """Insert term in descending exponent order"""
        if coefficient == 0:
            return
            
        new_node = Node(coefficient, exponent)
        
        # Empty list case
        if not self.head:
            self.head = self.tail = new_node
            self._length += 1
            return
            
        # Insert before head if exponent is largest
        if exponent > self.head.exponent:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
            self._length += 1
            return
            
        # Insert after tail if exponent is smallest
        if exponent < self.tail.exponent:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
            self._length += 1
            return
            
        # Find insertion point
        current = self.head
        while current and current.exponent > exponent:
            current = current.next
            
        # Skip if term already exists
        if current and current.exponent == exponent:
            current.coefficient += coefficient
            if current.coefficient == 0:
                self._remove_node(current)
            return
            
        # Insert before current
        new_node.next = current
        if current:  # If not inserting at end
            new_node.prev = current.prev
            current.prev.next = new_node
            current.prev = new_node
        else:  # Insert at end
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        self._length += 1

    def _remove_node(self, node):
        """Remove a node from the polynomial"""
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
            
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev
            
        self._length -= 1

    def evaluate(self, x):
        """Evaluate polynomial at x using Horner's method for efficiency"""
        result = 0
        current = self.head
        while current:
            result = result * (x ** (current.prev.exponent - current.exponent if current.prev else current.exponent)) + current.coefficient
            current = current.next
        return result

    def add(self, other):
        """Add two polynomials"""
        result = Polynomial()
        p1, p2 = self.head, other.head
        
        while p1 or p2:
            if not p2 or (p1 and p1.exponent > p2.exponent):
                result.insert_term(p1.coefficient, p1.exponent)
                p1 = p1.next
            elif not p1 or (p2 and p2.exponent > p1.exponent):
                result.insert_term(p2.coefficient, p2.exponent)
                p2 = p2.next
            else:
                coeff = p1.coefficient + p2.coefficient
                if coeff != 0:
                    result.insert_term(coeff, p1.exponent)
                p1, p2 = p1.next, p2.next
        return result

    def multiply(self, other):
        """Multiply two polynomials"""
        result = Polynomial()
        p1 = self.head
        while p1:
            p2 = other.head
            while p2:
                coeff = p1.coefficient * p2.coefficient
                exp = p1.exponent + p2.exponent
                result.insert_term(coeff, exp)
                p2 = p2.next
            p1 = p1.next
        return result

    def derivative(self):
        """Compute the derivative"""
        result = Polynomial()
        current = self.head
        while current:
            if current.exponent > 0:
                coeff = current.coefficient * current.exponent
                exp = current.exponent - 1
                result.insert_term(coeff, exp)
            current = current.next
        return result

    def integral(self, constant=0):
        """Compute the integral with integration constant"""
        result = Polynomial()
        result.insert_term(constant, 0)
        current = self.head
        while current:
            coeff = current.coefficient / (current.exponent + 1)
            exp = current.exponent + 1
            result.insert_term(coeff, exp)
            current = current.next
        return result

    def __str__(self):
        """String representation of the polynomial"""
        terms = []
        current = self.head
        while current:
            if current.exponent == 0:
                term = f"{current.coefficient:.2f}"
            elif current.exponent == 1:
                term = f"{current.coefficient:.2f}x"
            else:
                term = f"{current.coefficient:.2f}x^{current.exponent}"
            terms.append(term)
            current = current.next
        return " + ".join(terms) if terms else "0"

class PolynomialRegression:
    """Polynomial regression model using linked list implementation"""
    def __init__(self, degree=2, regularization=None, alpha=0.1):
        self.degree = degree
        self.regularization = regularization  # 'l1', 'l2', or None
        self.alpha = alpha
        self.poly = Polynomial()
        self.loss_history = []

    def fit(self, X, y, learning_rate=0.01, epochs=1000, verbose=False):
        """Train model using gradient descent"""
        # Initialize coefficients
        for d in range(self.degree + 1):
            self.poly.insert_term(np.random.normal(scale=0.1), d)
        
        for epoch in range(epochs):
            gradients = Polynomial()
            y_pred = np.array([self.poly.evaluate(x) for x in X])
            error = y_pred - y
            mse = np.mean(error**2)
            self.loss_history.append(mse)
            
            # Compute gradients for each term
            current = self.poly.head
            while current:
                grad = np.mean(error * (X ** current.exponent))
                
                # Add regularization
                if self.regularization == 'l1':
                    grad += self.alpha * np.sign(current.coefficient)
                elif self.regularization == 'l2':
                    grad += self.alpha * current.coefficient
                    
                gradients.insert_term(grad, current.exponent)
                current = current.next
            
            # Update coefficients
            current = self.poly.head
            grad_current = gradients.head
            while current and grad_current:
                current.coefficient -= learning_rate * grad_current.coefficient
                current = current.next
                grad_current = grad_current.next
                
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch}: MSE = {mse:.4f}")

    def predict(self, X):
        """Make predictions"""
        return np.array([self.poly.evaluate(x) for x in X])

class FeatureTransformer:
    """Polynomial feature engineering"""
    def __init__(self, degree=2, include_interactions=True):
        self.degree = degree
        self.include_interactions = include_interactions
        
    def transform(self, X):
        """Generate polynomial features"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        n_samples, n_features = X.shape
        features = []
        
        # Add polynomial terms for each feature
        for i in range(n_features):
            for d in range(1, self.degree + 1):
                features.append(X[:, i] ** d)
                
        # Add interaction terms if enabled
        if self.include_interactions and n_features > 1:
            for i in range(n_features):
                for j in range(i+1, n_features):
                    features.append(X[:, i] * X[:, j])
                    
        return np.column_stack(features)

class PolynomialVisualizer:
    """Visualization utilities for polynomials"""
    @staticmethod
    def plot_polynomial(poly, x_range=(-5, 5), num_points=500, title=None):
        """Plot polynomial function"""
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = [poly.evaluate(xi) for xi in x]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, linewidth=2)
        plt.title(title or f"Polynomial: {str(poly)}")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.grid(True, alpha=0.3)
        plt.show()
        
    @staticmethod
    def plot_loss(loss_history, log_scale=False):
        """Plot training loss curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        if log_scale:
            plt.yscale('log')
        plt.title("Training Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        plt.grid(True, alpha=0.3)
        plt.show()

# ======================
# Demonstration Examples
# ======================

def demo_polynomial_operations():
    print("\n=== Polynomial Operations Demo ===")
    p1 = Polynomial()
    p1.insert_term(3, 2)  # 3x^2
    p1.insert_term(-2, 1) # -2x
    p1.insert_term(5, 0)  # 5
    
    p2 = Polynomial()
    p2.insert_term(1, 3)  # x^3
    p2.insert_term(4, 1)  # 4x
    
    print(f"p1 = {p1}")
    print(f"p2 = {p2}")
    
    # Addition
    p_add = p1.add(p2)
    print(f"\np1 + p2 = {p_add}")
    
    # Multiplication
    p_mult = p1.multiply(p2)
    print(f"\np1 * p2 = {p_mult}")
    
    # Derivative
    p_deriv = p1.derivative()
    print(f"\nd/dx(p1) = {p_deriv}")
    
    # Integral
    p_integral = p1.integral()
    print(f"\n∫p1 dx = {p_integral} + C")
    
    # Visualization
    PolynomialVisualizer.plot_polynomial(p1, title="Example Polynomial")

def demo_regression():
    print("\n=== Polynomial Regression Demo ===")
    np.random.seed(42)
    X = np.linspace(-3, 3, 100)
    y = 0.5 * X**3 - 2 * X**2 + X + np.random.normal(0, 1, 100)
    
    model = PolynomialRegression(degree=3)
    model.fit(X, y, learning_rate=0.001, epochs=5000, verbose=True)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, s=10, label="Data")
    X_sorted = np.sort(X)
    plt.plot(X_sorted, model.predict(X_sorted), 'r-', label="Polynomial Fit")
    plt.title(f"Polynomial Regression Fit: {model.poly}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Plot loss history
    PolynomialVisualizer.plot_loss(model.loss_history, log_scale=True)

def demo_feature_engineering():
    print("\n=== Feature Engineering Demo ===")
    X = np.array([[1, 2], [3, 4], [5, 6]])
    print("Original features:")
    print(X)
    
    transformer = FeatureTransformer(degree=2, include_interactions=True)
    X_poly = transformer.transform(X)
    
    print("\nPolynomial features:")
    print(X_poly)

if __name__ == "__main__":
    demo_polynomial_operations()
    demo_regression()
    demo_feature_engineering()
