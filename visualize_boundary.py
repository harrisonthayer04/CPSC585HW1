import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns

def load_data(filename):
    data = []
    targets = []
    with open(filename, 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split(',')]
            data.append(values[:-1])  # all but last value
            targets.append(values[-1])  # last value is target
    return np.array(data), np.array(targets)

def load_weights(filename):
    weights = []
    with open('weight_evolution.txt', 'r') as f:
        for line in f:
            w = [float(x) for x in line.strip().split(',') if x]
            weights.append(w)
    return np.array(weights)[-1]  # Return only final weights

def plot_pie_charts(y_train, y_test):
    plt.figure(figsize=(10, 4))
    
    # Training data pie chart
    plt.subplot(1, 2, 1)
    train_graduates = np.sum(y_train == 1)
    train_dropouts = np.sum(y_train == 0)
    plt.pie([train_graduates, train_dropouts], 
            labels=['Graduates', 'Dropouts'],
            autopct='%1.1f%%',
            colors=['blue', 'red'])
    plt.title('Training Data Distribution')
    
    # Test data pie chart
    plt.subplot(1, 2, 2)
    test_graduates = np.sum(y_test == 1)
    test_dropouts = np.sum(y_test == 0)
    plt.pie([test_graduates, test_dropouts], 
            labels=['Graduates', 'Dropouts'],
            autopct='%1.1f%%',
            colors=['blue', 'red'])
    plt.title('Test Data Distribution')
    
    plt.tight_layout()
    plt.show()

def plot_decision_boundary(X_train, y_train, X_test, y_test, weights):
    plt.figure(figsize=(12, 5))
    
    # Plot training data
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[y_train==0][:, 0], X_train[y_train==0][:, 1], 
               c='red', label='Dropouts', alpha=0.6)
    plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], 
               c='blue', label='Graduates', alpha=0.6)
    
    # Calculate and plot decision boundary line
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    x = np.array([x_min, x_max])
    y = (-weights[0]*x)/weights[1]
    plt.plot(x, y, 'g-', label='Decision Boundary')
    
    plt.title('Training Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    
    # Plot test data
    plt.subplot(1, 2, 2)
    plt.scatter(X_test[y_test==0][:, 0], X_test[y_test==0][:, 1], 
               c='red', label='Dropouts', alpha=0.6)
    plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], 
               c='blue', label='Graduates', alpha=0.6)
    
    # Plot same decision boundary
    plt.plot(x, y, 'g-', label='Decision Boundary')
    
    plt.title('Test Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def main():
    # Load training and test data
    X_train, y_train = load_data('training_data.txt')
    X_test, y_test = load_data('test_data.txt')
    
    # Plot pie charts for data distribution
    plot_pie_charts(y_train, y_test)
    
    # Use PCA to reduce to 2 dimensions
    global pca  # Make pca accessible to plot_confusion_matrices
    pca = PCA(n_components=2)
    X_train_2d = pca.fit_transform(X_train)
    X_test_2d = pca.transform(X_test)
    
    # Load final weights
    weights = load_weights('weight_evolution.txt')
    w_2d = pca.transform(weights.reshape(1, -1))[0]
    
    # Plot decision boundary
    plot_decision_boundary(X_train_2d, y_train, X_test_2d, y_test, w_2d)

if __name__ == "__main__":
    main() 