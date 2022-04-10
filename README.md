# SRIP-IITGN-2022
SRIP IITGN 2022 - Machine Learning - Prof. Nipun Batra

### JAX
Jax is a Python library designed for high-performance ML research. Jax is nothing more than a numerical computing library, just like Numpy, but with some key improvements. It was developed by Google and used internally both by Google and Deepmind teams.
References
 * [JAX Quickstart](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)
 * [Youtube Playlist](https://youtube.com/playlist?list=PLBoQnSflObckOARbMK9Lt98Id0AKcZurq)

### Animate bivariate normal distribution
It is a special case, multi-variate distribution. It's parameters are the mean vector which will have 2 elements and a covariance matrix.
[Check the implementation here](https://github.com/Khushm/SRIP-IITGN-2022/blob/main/Animate%20Bivariate%20Normal%20Distribution.ipynb)

References
  * [Blog - Gaussian Processes](https://nipunbatra.github.io/blog/ml/2019/08/20/Gaussian-Processes.html)
  * [Blog - 3D AND CONTOUR PLOTS OF THE BIVARIATE NORMAL DISTRIBUTION](https://datasciencegenie.com/3d-contour-plots-of-bivariate-normal-distribution/)
  * [Video - Draw samples from a multivariate normal using numpy and scipy](https://www.youtube.com/watch?v=ppd4c96hHH8)
  * [Blog - A Tutorial on Generating & Plotting 3D Gaussian Distributions with (Python/Numpy/Tensorflow/Pytorch) & (Matplotlib/Plotly)](https://towardsdatascience.com/a-python-tutorial-on-generating-and-plotting-a-3d-guassian-distribution-8c6ec6c41d03)

### Implement from scratch a sampling method to draw samples from a multivariate Normal (MVN) distribution
The multivariate normal distribution is a multidimensional generalisation of the one-dimensional normal distribution . It represents the distribution of a multivariate random variable that is made up of multiple random variables that can be correlated with each other.

References
 * [Eric's Notes - ESTIMATING A MULTIVARIATE GAUSSIAN'S PARAMETERS BY GRADIENT DESCENT](https://ericmjl.github.io/notes/stats-ml/estimating-a-multivariate-gaussians-parameters-by-gradient-descent/)
 * [Blog - Sampling from a Multivariate Normal Distribution](https://juanitorduz.github.io/multivariate_normal/)
 
### Implement two hidden layers neural network classifier from scratch
We will create a neural network with one input layer, two hidden layer, and one output layer. We have a neural network with 784 inputs, the first hidden layer has 512 nodes while the second hidden layer consist of 256 nodes. The output layer has 10 node since we are solving a mnist classification problem, where there can be only 10 possible outputs. 
[Check the implementation here](https://github.com/Khushm/SRIP-IITGN-2022/blob/main/Neural%20Network%20Classifier.ipynb)

References
 * [Blog - Build Neural Network from Scratch](https://towardsdatascience.com/how-to-build-neural-network-from-scratch-d202b13d52c1)
 * [Blog - Creating a Neural Network from Scratch in Python](https://stackabuse.com/creating-a-neural-network-from-scratch-in-python-adding-hidden-layers/)
 * [Article - Neural Networks](https://developer.ibm.com/articles/neural-networks-from-scratch/)
 * [Documentation - NN with JAX](https://jax.readthedocs.io/en/latest/notebooks/neural_network_with_tfds_data.html)

### Bayesian Linear Regression from scratch
The Bayesian approach describes probability as a measurement of belief in an event, using the prior and acquired knowledge from observed data. The goal is to update the probability distributions of the parameters by incorporating information about the parameters from observing the data. 

References
 * [Blog - Bayesian Linear Regression](https://nipunbatra.github.io/blog/ml/2020/02/20/bayesian-linear-regression.html)
 * [Blog - Implementing Bayesian Linear Regression](https://towardsdatascience.com/implementing-bayesian-linear-regression-9375a9994f98)
 * [Video - Machine Learning: Bayesian Linear Regression](https://www.youtube.com/watch?v=LzZ5b3wdZQk)
 * [BlackJAX - Bayesian Linear Regression](https://blackjax-devs.github.io/blackjax/examples/LogisticRegression.html)

