# C++ Model Predictive Control (MPC) Simulation
This is C++ Simple implementation of MPC.
it aims to control a robot to follow a given path (circle, hallway, sine wave)
![MPC Simulation](./asset/sin.gif)

## Dependencies
* C++14 or higher (not tested lower compiler)
* CMake 3.14 or higher
* Eigen3
* Python2 or Python3 (for visualization)
	* numpy, matplotlib
* [matplotlib_cpp](https://github.com/lava/matplotlib-cpp)
	* the simulation requires a change in matplotlibcpp.h [[subplot issue]](https://stackoverflow.com/questions/70076843/the-matplotlibcpp-show-an-error-when-i-use-subplot-in-cpp)
## Build

```bash
git clone https://github.com/mkyun2/Simple_MPC.git
cd Simple_MPC

mkdir build
cd build

cmake ..
make or cmake --build .
```

## Run
```bash
# run simulation with circular path
./simulate circle

# run simulation with hallway path
./simulate hallway

# run simulation with sine wave path
./simulate sin
```
## MPC Problem
1. Model
$\text{Differential Wheeled Robot}$
$\begin{aligned}
\dot{x}&= v\ cos\theta \\
\dot{y}&= v\ sin\theta \\
\dot{\theta}&= \omega 
\end{aligned}$

2. Prediction
$\begin{aligned}
x_{k+1} &= A_kx_{k}+B_ku_{k} \\
x_{k+2} &= A_{k+1}x_{k+1}+B_{k+1}u_{k+1} \\
&=A_{k+1}A_{k}x_{k}+A_{k+1}B_{k}u_{k}+B_{k+1}u_{k+1} \\
x_{k+N} &= ... \\
x_{k+1:k+N} &= Fx_{k}+\Phi u_{k:k+N-1}
\end{aligned}$

3. Optimization
$\begin{aligned}
\min_{x, u} \quad & \sum_{k=0}^{N-1} (x_{k+1}^T Q x_{k+1} + u_k^T R u_k) \\
\text{subject to} \quad & A_k x_k + B_k u_K \\
& C_u u_k \leq d_u \\
& C_x x_k \leq d_x \text{(To do)} \\
\end{aligned}$

By substituting the system model constraint into the cost function, we transforms it into a problem terms of the control input sequence $U$.

$\begin{aligned}
\min_{U} \quad &\frac{1}{2}U^T\mathbf{H}U+\mathbf{f}^TU \\
&C_u U\leq d_u \\
\end{aligned}$


