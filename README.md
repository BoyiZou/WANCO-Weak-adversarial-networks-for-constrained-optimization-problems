# WANCO: Weak adversarial networks for constrained optimization problems

## A brief introduction to WANCO
Weak adversarial networks for constrained optimization problems (WANCO) is a deep learning based framework for constrained optimization problems related to differential operators. It integrates the augmented Lagrangian method and adversarial network to deal with such problems. More specifically, we first transform them into minimax problems using the augmented Lagrangian method and then use two (or several) deep neural networks(DNNs) to represent the primal and dual variables respectively. The parameters in the neural networks are then trained by an adversarial process.

The proposed architecture is relatively insensitive to the scale of values of different constraints when compared to penalty based deep learning methods. Extensive
examples for optimization problems with scalar constraints, nonlinear constraints, partial differential equation constraints, and inequality constraints are considered to show the capability and robustness of the proposed method, with applications ranging from Ginzburg–Landau energy minimization problems, partition problems, fluid-solid topology optimization, to obstacle problems.

## Paper
This repository contains the code for examples in the paper WANCO: Weak adversarial networks for constrained optimization problems by Gang Bao, Dong Wang, and Boyi Zou.
