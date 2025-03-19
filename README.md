# TKDEA

> This repository contains materials related to our paper published in TCYB 2024.

## Introduction

Bilevel optimization is a special type of optimization in which one problem is embedded within another. The bilevel optimization problem of which both levels are multiobjective functions is usually called the multiobjective bilevel optimization problem. The expensive computation and nested features make it challenging to solve. 

Under a limited computational budget, instead of always searching for the complete lower-level Pareto front, we focus on exploring the preferred solution w.r.t. the lower-level decision maker. We propose a multiobjective bilevel optimization solving routine combined with a knee point driven algorithm. Specifically, the proposed algorithm aims to quickly find feasible solutions considering the lower-level constraints in the first stage. Then it concentrates the computational resources on finding solutions with better performance. 

Besides, we develop several multiobjective bilevel test problems with different properties, such as scalable, deceptive, convexity, and (dis)continuous.  

Finally, the performance of the algorithm is validated on a practical petroleum refining bilevel problem, which involves a multiobjective environmental regulation problem and a petroleum refining operational problem.

![petroleum refining operational problem](PracticalProblem.png)
*Petroleum Refining Operational Problem*


## Contact

If you meet any problems, please feel free to contact me.

- Jiaxin Chen (chenjiaxin1934@gmail.com)
