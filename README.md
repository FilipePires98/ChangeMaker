# ChangeMaker
A study on ’The Change-Making Problem’

![](https://img.shields.io/badge/Academical%20Project-Yes-success)
![](https://img.shields.io/badge/Made%20With-Python-blue)
![](https://img.shields.io/badge/License-Free%20To%20Use-green)
![](https://img.shields.io/badge/Maintained-No-red)

# Description

It is widely known that many computationally-demanding problems require more efficient solutions than to simply search exaustively for the answers.
This is particularly true for tasks that execute repeated operations in considerable amounts of times. 

This project focuses on the famous challenge called ’The Change-Making Problem’ and presents a study on the computational performance of dynamic programming algorithms for solving such problem.

The implemented algorithms were 3, and they all follow the same base strategy to tackle the issue so that a high fidelity comparison could be analysed.

## Repository Structure

/report - documentation on the conducted study

/results - outputs produced by the implemented code

/src - source code of the algorithms 

## Data Visualization

<img src="https://github.com/FilipePires98/ChangeMaker/blob/main/results/elaborate/tCMP_elaborate_results_plot_%5B1%205%5D.png" width="540px">

Plot presenting the evolution of the (average) execution times of the 3 algorithms for the same input change amount, suggesting that the dynamic-programming-based implementation is for efficient that the others.

## Instructions to Run

```console
$ cd src
$ pip3 install -r requirements.txt
$ python3 TheChangeMakingProblem.py
```

## Author

The author of this repository is Filipe Pires, and the project was developed for the Advanced Algorithms Course of the master's degree in Informatics Engineering of the University of Aveiro.

For further information read the [report](https://github.com/FilipePires98/ChangeMaker/blob/main/report/report.pdf) or contact me at filipesnetopires@ua.pt.
