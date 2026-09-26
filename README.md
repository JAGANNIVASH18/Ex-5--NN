<H3>Name :JAGANNIVASH U M</H3>
<H3>Register No : 212224240059</H3>
<H3>EX. NO.5</H3>
<H3>Date : 26-08-2026</H3>
<H1 ALIGN =CENTER>Implementation of XOR  using RBF</H1>
<H3>Aim:</H3>
To implement a XOR gate classification using Radial Basis Function  Neural Network.

<H3>Theory:</H3>
<P>Exclusive or is a logical operation that outputs true when the inputs differ.For the XOR gate, the TRUTH table will be as follows XOR truth table </P>

<P>XOR is a classification problem, as it renders binary distinct outputs. If we plot the INPUTS vs OUTPUTS for the XOR gate, as shown in figure below </P>




<P>The graph plots the two inputs corresponding to their output. Visualizing this plot, we can see that it is impossible to separate the different outputs (1 and 0) using a linear equation.
A Radial Basis Function Network (RBFN) is a particular type of neural network. The RBFN approach is more intuitive than MLP. An RBFN performs classification by measuring the input’s similarity to examples from the training set. Each RBFN neuron stores a “prototype”, which is just one of the examples from the training set. When we want to classify a new input, each neuron computes the Euclidean distance between the input and its prototype. Thus, if the input more closely resembles the class A prototypes than the class B prototypes, it is classified as class A ,else class B.
A Neural network with input layer, one hidden layer with Radial Basis function and a single node output layer (as shown in figure below) will be able to classify the binary data according to XOR output.
</P>





<H3>ALGORITHM:</H3>
Step 1: Initialize the input  vector for you bit binary data<Br>
Step 2: Initialize the centers for two hidden neurons in hidden layer<Br>
Step 3: Define the non- linear function for the hidden neurons using Gaussian RBF<br>
Step 4: Initialize the weights for the hidden neuron <br>
Step 5 : Determine the output  function as 
                 Y=W1*φ1 +W1 *φ2 <br>
Step 6: Test the network for accuracy<br>
Step 7: Plot the Input space and Hidden space of RBF NN for XOR classification.

<H3>PROGRAM:</H3>

```py
import numpy as np
import matplotlib.pyplot as plt

def gaussian_rbf(x, landmark, gamma=1):
    return np.exp(-gamma * np.linalg.norm(x - landmark) ** 2)

def end_to_end(X1, X2, ys, mu1, mu2):
    X = np.column_stack((X1, X2))

    from_1 = np.array([gaussian_rbf(x, mu1) for x in X])
    from_2 = np.array([gaussian_rbf(x, mu2) for x in X])

    A = np.column_stack((from_1, from_2))

    weights = np.linalg.pinv(A).dot(ys)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter((X1[0], X1[3]), (X2[0], X2[3]), label="Class 0")
    plt.scatter((X1[1], X1[2]), (X2[1], X2[2]), label="Class 1")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("XOR: Linearly Inseparable")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(from_1[[0, 3]], from_2[[0, 3]], label="Class 0")
    plt.scatter(from_1[[1, 2]], from_2[[1, 2]], label="Class 1")
    plt.xlabel("RBF1")
    plt.ylabel("RBF2")
    plt.title("RBF Hidden Space")
    plt.legend()

    plt.show()

    return weights

def predict_matrix(point, weights):
    mu1 = np.array([0, 1])
    mu2 = np.array([1, 0])

    phi1 = gaussian_rbf(point, mu1)
    phi2 = gaussian_rbf(point, mu2)

    output = np.array([phi1, phi2]).dot(weights)

    return np.round(output)

x1 = np.array([0, 0, 1, 1])
x2 = np.array([0, 1, 0, 1])
ys = np.array([0, 1, 1, 0])

mu1 = np.array([0, 1])
mu2 = np.array([1, 0])

weights = end_to_end(x1, x2, ys, mu1, mu2)

print("XOR Classification Results")
print("---------------------------")

for point in [
    np.array([0, 0]),
    np.array([0, 1]),
    np.array([1, 0]),
    np.array([1, 1])
]:
    prediction = predict_matrix(point, weights)
    print("Input:", point, "Predicted Output:", prediction)

inputs = [
    np.array([0, 0]),
    np.array([0, 1]),
    np.array([1, 0]),
    np.array([1, 1])
]

predictions = np.array([
    predict_matrix(point, weights)
    for point in inputs
])

accuracy = np.mean(predictions == ys) * 100

print("\nExpected Output:", ys)
print("Predicted Output:", predictions)
print("Accuracy:", accuracy, "%")
```

<H3>OUTPUT:</H3>

<img width="1130" height="676" alt="image" src="https://github.com/user-attachments/assets/16e5b5a1-fd96-48b8-b457-78650b367f38" />


<H3>Result:</H3>
Thus , a Radial Basis Function Neural Network is implemented to classify XOR data.








