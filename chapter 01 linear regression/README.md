# Linear Regression

## 回归

回归是基于给定的特征值，预测目标变量值的过程。数学上，回归旨在构建一个函数$f$来建立输入数据$x={(x_1,x_2,\cdots,x_m)}$和监督值$y$的联系。此后，我们便可以使用$f$来预测$x$对应的监督值$\hat{y}$，即
$$
\hat{y}=f(x)
$$
## 线性回归

函数$f$的形式是线性的，即$f$具有如下形式
$$
f(x)=f(x_1,x_2,\cdots,x_m)=w_0+w_1x_1+w_2x_2+\cdots+w_mx_m
$$
## 回归的目标

对于给定的训练数据集
$$
\{(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),\cdots,(x^{(n)},y^{(n)})\}
$$
找到参数$\{w_0,w_1,\cdots,w_m\}$，使得预测值$\hat{y}=f(x)$可以和真实值$y$尽可能地接近。

## 损失函数

为了刻画训练集上的预测值$(\hat{y}_1,\hat{y}_2,\cdots,\hat{y}_n)$和真实值$(y_1,y_2,\cdots,y_n)$的接近程度，我们引入损失函数来刻画这种接近程度。

## 多特征线性回归

一般地，线性回归写成如下形式
$$
f(x_1,x_2,\cdots,x_m)=w_0+w_1x_1+w_2x_2+\cdots+w_mx_m
$$
其中，$x_i$表示数据点的第$i$个特征。

引入矩阵后，我们的表示如下
$$
f({\bf{x}})=\bf{xw}
$$

$$
{\bf{x}}=[1, x_1, x_2,\cdots,x_m]
$$

$$
{\bf{w}}=[w_0, w_1,w_2,\cdots,w_m]^T
$$

若选择均方误差为损失函数，则损失函数可表示为
$$
L({\bf{w}})=\frac{1}{n}\sum_{i=1}^n({\bf{x}}^{(i)}{\bf{w}}-y^{(i)})^2
$$
引入向量的模，则上述公式可表示为
$$
L({\bf{w}})=\frac{1}{n}||{\bf{Xw}-y}||^2
$$

$$
{\bf{X}}=
\begin{bmatrix}
&{\bf{x}}^{(1)}\\
&{\bf{x}}^{(2)}\\
&\vdots\\
&{\bf{x}}^{(n)}
\end{bmatrix},\ {\bf{y}}=
\begin{bmatrix}
&y^{(1)}\\
&y^{(2)}\\
&\vdots\\
&y^{(n)}
\end{bmatrix}
$$

## 梯度

梯度是一个向量，函数沿着梯度的方向增加最快。即沿着梯度方向找最大值，逆梯度方向找最小值。一般地，记梯度算子为$\nabla$，则有

对于以$n\times1$实向量$\bf{x}$为变量的实标量函数$f({\bf{x}})$相对于${\bf{x}}$的梯度为$1\times n$的列向量
$$
\nabla_{\bf{x}}f({\bf{x}})=\begin{bmatrix}
\frac{\partial f}{\partial x_1}\\
\frac{\partial f}{\partial x_2}\\
\vdots\\
\frac{\partial f}{\partial x_n}\\
\end{bmatrix}
$$
对于$m$维行向量函数${\bf{f}}({\bf{x}})=[{\bf{f}}_1({\bf{x}}),{\bf{f}}_2({\bf{x}}),\cdots,{\bf{f}}_m({\bf{x}})]$相对于$n\times 1$的实向量$\bf{x}$的梯度是一个$n\times m$的矩阵。
$$
\nabla_{\bf{x}}{\bf{f}}({\bf{x}})=\begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_2}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_1}\\
\frac{\partial f_1}{\partial x_2} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_2}\\
\vdots & \vdots & \ddots & \vdots\\
\frac{\partial f_1}{\partial x_n} & \frac{\partial f_2}{\partial x_n} & \cdots & \frac{\partial f_m}{\partial x_n}\\
\end{bmatrix}
$$
## 均方误差的最优值

损失函数的梯度为
$$
\frac{\partial L({\bf{w}})}{{\partial\bf{w}}}=\frac{2}{n}{\bf{X}}^T({\bf{Xw-y}})
$$

由于$L({\bf{w}})$是凸函数，最优解由下列方程给出
$$
{\bf{X}}^T({\bf{Xw-y}})=0
$$
解得
$$
{\bf{w}}^*=({\bf{X}^TX})^{-1}{\bf{X}}^T{\bf{}y}
$$

## 梯度下降法

在大多数情况下，分析解并不都是存在的，或者计算出分析解的代价太大，因此我们需要将求解过程转向数值解。此时，我们采用梯度下降法求解，其更新公式为
$$
\left.{\bf{w}}^{(t+1)}={\bf{w}}^{(t)}-r\cdot\frac{\partial L({\bf{w}})}{\partial {\bf{w}}}\right|_{{\bf{w}}={\bf{w}}^{(t)}}
$$
其中，$r$被称为学习率，是人为指定的。当$r$过大时，会出现发散的情况；当$r$过小时，会出现收敛过慢的情况。

## 随机梯度下降

当训练数据过大时，梯度下降法计算过程慢，SGD则改为使用小批量的数据来计算梯度。

## 均方误差的最优值的推导：

损失函数的公式为
$$
L({\bf{w}})=\frac{1}{n}||{\bf{Xw}-y}||^2
$$
我们需要计算梯度
$$
\nabla L({\bf{w}})=\left[\frac{\partial L({\bf{w}})}{\partial w_0}\ \frac{\partial L({\bf{w}})}{\partial w_1}\ \cdots\ \frac{\partial L({\bf{w}})}{\partial w_m}\right]^T
$$
将$L({\bf{w}})$展开得
$$
\begin{align}
L({\bf{w}})
&=\frac{1}{n}\sum_{i=1}^n({\bf{x}}^{(i)}{\bf{w}}-y^{(i)})^2\\
&=\frac{1}{n}\sum_{i=1}^n(w_0+w_1x_1^{(i)}+\cdots+w_mx_m^{(i)}-y^{(i)})^2
\end{align}
$$
对${\bf{w}}$其中一个分量$w_j$求导得
$$
\begin{align}
\frac{\partial L({\bf{w}})}{w_j}
&=\frac{1}{n}\sum_{i=1}^n2({\bf{x}}^{(i)}{\bf{w}}-{{y}}^{(i)})x_j^{(i)}\\
&=\frac{2}{n}[x_j^{(1)}({\bf{x}}^{(1)}{\bf{w}}-{{y}}^{(1)})+x_j^{(2)}({\bf{x}}^{(2)}{\bf{w}}-{y}^{(2)})+\cdots+x_j^{(n)}({\bf{x}}^{(n)}{\bf{w}}-{y}^{(n)})]\\
&=\frac{2}{n}[x_j^{(1)}\ x_j^{(2)}\ \cdots x_j^{(n)}]
\begin{bmatrix}
{\bf{x}}^{(1)}{\bf{w}}-{y^{(1)}}\\
{\bf{x}}^{(2)}{\bf{w}}-{y^{(2)}}\\
\vdots\\
{\bf{x}}^{(n)}{\bf{w}}-{y^{(n)}}
\end{bmatrix}\\
&=\frac{2}{n}{\bf{X}}_{[:,j]}^T({\bf{Xw-y}})
\end{align}
$$
其中，${\bf{X}}_{[:,j]}$表示矩阵${\bf{X}}$的第$j$列。将上式代入$\nabla f(\bf{w})$可得
$$
\begin{align}
\nabla f({\bf{x}})
&=\left[\frac{2}{n}{\bf{X}}_{[:,0]}^T({\bf{Xw-y}})\ \frac{2}{n}{\bf{X}}_{[:,1]}^T({\bf{Xw-y}})\ \cdots\ \frac{2}{n}{\bf{X}}_{[:,m]}^T({\bf{Xw-y}})\right]^T\\
&=\frac{2}{n}
\begin{bmatrix}{\bf{X}}_{[:,0]}^T\\
{\bf{X}}_{[:,1]}^T\\
\vdots\\
{\bf{X}}_{[:,m]}^T
\end{bmatrix}({\bf{Xw-y}})\\
&=\frac{2}{n}{\bf{X}}^T({\bf{Xw-y}})
\end{align}
$$
