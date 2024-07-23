# WANCO: Weak adversarial networks for constrained optimization problems

## A brief introduction to WANCO
* Weak adversarial networks for constrained optimization problems (WANCO) is a deep learning based framework for constrained optimization problems related to differential operators. It integrates the augmented Lagrangian method and adversarial network to deal with such problems. More specifically, we first transform them into minimax problems using the augmented Lagrangian method and then use two (or several) deep neural networks(DNNs) to represent the primal and dual variables respectively. The parameters in the neural networks are then trained by an adversarial process.

* The proposed architecture is relatively insensitive to the scale of values of different constraints when compared to penalty based deep learning methods. Extensive
examples for optimization problems with scalar constraints, nonlinear constraints, partial differential equation constraints, and inequality constraints are considered to show the capability and robustness of the proposed method, with applications ranging from Ginzburg–Landau energy minimization problems, partition problems, fluid-solid topology optimization, to obstacle problems.

## Paper
* This repository contains the code in the paper [WANCO: Weak adversarial networks for constrained optimization problems](https://arxiv.org/abs/2407.03647) by Gang Bao, Dong Wang, and Boyi Zou.

## About the code
* The code in this repository was written by '[Python 3.10](https://www.python.org/downloads/)' and 'PyTorch 2.3.1'. (or it can be directly run on Colab)
* Here we provide the code from the article and have added comments to some of them. (An example of each type of numerical example has been commented, and the rest are similar. See below for details.)
  * The code for the Ginsburg-Landau model is provided in 'Ginzburg Landau', where 'WANCO1q' provides some code annotations.
  * The code for the activation test is provided in 'Activation test'.
  * The code for the Dirichlet partition problem is provided in 'Dirichlet partition', where 'PBC-2d-n=25' provides some code annotations.
  * The code for the fluid-solid optimization problem is provided in 'Fluid-solid optimization', where 'case2_d=0_5' provides some code annotations.
  * The code for the Obstacle problem is provided in 'Obstacle problem', where 'obstacle1' provides some code annotations.
* One should keep in mind that the parameters provided in this code may not efficiently work for different types of problems. So one may need to readjust parameters when using this code for solving different problems.
* In this work, networks used are basically ResNet combined with activation $\tanh^3(\cdot)$, and the effectiveness is verified through numerical experiments. Since WANCO is a framework algorithm, it can naturally be combined with other network structures or integrated with other deep learning methods to further enhance the algorithm's capability and stability to other constrained optimization problems. 
