# 二分类问题

**Sigmoid/logistic函数**：
$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$
通过sigmoid函数，我们可以将线性回归的值域从$[-\infty,+\infty]$变换到$[0,1]$，使得我们可以用$0,1$来表示类别。

此时，Logistic回归如下。

**Logistic regression**：
$$
f({\bf x})=\sigma({\bf xw})
$$
**cost function**：我们的目标是当数据的类别$y=1$时，我们希望$f({\bf x})$尽可能的接近1；当当数据的类别$y=0$时，我们希望$f({\bf x})$尽可能的接近0。

我们可以定义均方误差的损失函数如下。
$$
L({\bf w})=(\sigma({\bf xw})-y)^2
$$
当更常用的是交叉熵损失函数（cross-entropy loss）。
$$
L({\bf w})=
\begin{cases}
& -\log(\sigma({\bf xw})),\ y=1\\
& -\log(1-\sigma({\bf xw})),\ y=0
\end{cases}
$$
可以写成
$$
L({\bf w})=-y\log(\sigma({\bf xw}))-(1-y)\log(1-\sigma({\bf xw}))
$$
交叉熵是凸函数，而均方误差不是，因此交叉熵可以保证收敛到最优值。

此时，交叉熵损失的梯度为
$$
\frac{\partial L({\bf w})}{\partial {\bf w}}=\frac{1}{N}\sum_{i=1}^N[\sigma({\bf x}^{(i)}{\bf w})-y^{(i)}]{\bf x}^{(i)T}
$$
可以令
$$
\frac{\partial L({\bf w})}{\partial {\bf w}}=0
$$
来求出最优值，但这里的解析解不存在。由于交叉熵损失是凸函数，因此可以使用梯度下降的方式来求解。
$$
{\bf w}^{(t+1)}={\bf w}^{(t)}-r\cdot\frac{\partial L({\bf w})}{\partial {\bf w}}
$$
**决策边界（decision boundary）**：分类的依据可以如下所示
$$
\hat{y}=\begin{cases}
& 1,\ \sigma({\bf xw})\ge 0.5\\
& 0,\ \sigma({\bf xw}) < 0.5
\end{cases}
$$
等价于
$$
\hat{y}=\begin{cases}
& 1,\ {\bf xw}\ge 0\\
& 0,\ {\bf xw} < 0
\end{cases}
$$
因此，决策边界是满足方程
$$
{\bf xw}=0
$$
的${\bf x}$构成的超平面，是线性的决策边界。

# 多分类问题

假设需要分类的问题有$K$类，在二分类的基础上，对每一个类别引入组权值即可。

**softmax函数**：
$$
\text{softmax}_i({\bf z})=\frac{e^{z_i}}{\sum_{k=1}^Ke^{z_k}},\ i=1,2,\cdots,K
$$
在多分类问题中，数据点${\bf x}$属于第$i$类的概率是
$$
f_i({\bf x})=\text{softmax}_i({\bf xW})=\frac{e^{\bf xw_i}}{\sum_{k=1}^Ke^{\bf xw_k}}
$$
其中，$\bf W=[w_1,w_2,\cdots,w_K]$

在二分类问题上，softmax函数等价于逻辑函数。

**损失函数**：对于多分类问题，标记$\bf y$使用one-hot向量来表示，即
$$
[1,0,0,\cdots,0],\\
[0,1,0,\cdots,0],\\
\vdots\\
[0,0,0,\cdots,1],\\
$$
训练的目标是最大化训练样本对应的分类概率，即
$$
L({\bf w_1},{\bf w_2},\cdots,{\bf w_K})=-\frac{1}{N}\sum_{i=1}^N\sum_{k=1}^Ky_k^{(i)}\log(\text{softmax}_k({\bf x}^{(i)}{\bf W}))
$$
此时可以得到梯度为
$$
\frac{\partial L({\bf W})}{\partial {\bf W}}=\frac{1}{N}\sum_{i=1}^N[\sigma({\bf x}^{(i)}{\bf W})-{\bf y}^{(i)}]{\bf x}^{(i)T}
$$
使用梯度下降法更新
$$
\left .{\bf W}_{t+1}={\bf W}_t-r\cdot\frac{\partial L(\bf W)}{\partial \bf W}\right|_{\bf w=w_t}
$$
