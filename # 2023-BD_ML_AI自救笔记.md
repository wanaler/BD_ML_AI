# 2023-BD_ML_AI自救笔记
##### 这个笔记旨在用互助的方式帮助可怜的能动孩子理解Noack教授的课堂内容
---
## Lecture 1 
### Proximity map
* #### 简要介绍(临近图)
    对于高维空间的中的数据点（例如xyz坐标中的一个点），要想映射到低维空间（xy坐标系），需要通过数据降维，一些降维方法保留了中心点与临近点之间的相对关系，使得高位空间的部分信息能够在低维空间中得以体现，这样在低维空间中画出来的图便是Proximity map(临近图)
### Locally linear embedding(局部线性插值)
* #### 内容参考

    [[BiliBili] python与人工智能-数据降维-LLE](https://www.bilibili.com/video/BV14g411F7Je/?spm_id_from=333.337.search-card.all.click&vd_source=35bbad48bf3d84d0d268e71078d2cf2e)
* #### 简要介绍
    **"Locally Linear Embedding"** 下面简称 **LLE** 是一种实现数据降维的方法。简单来说，对于一个高维空间的数据点，映射到低维空间中时，要想保持与其他点相对位置的一致性，**可以使用LLE方法**,如下图所示：
    ![LLE](/notebook/LLE_1.png)
<center> LLE原理</center>

* #### 具体实现
    如上图，将高维空间的一个点 $X_i$ 映射到低维空间中（ $X_i$ 可以看成一个向量，其中的每一个元素可以看成 $X_i$ 的属性），可以选取 $X_i$ 周围的的点（临近点 $X_j$ ），计算出临近点与中心点 $X_i$的相对为位置关系（即 $W_{ij}$ , 表示 $X_i$ 与 $X_j$ 之间的位置关系），在映射的时候，保证 $X_i$的映射对应点 $Y_i$ 与周围的临近点 $Y_j$ 保持类似的位置关系（ $W_{ij}$ ）就可以了。

    1.构造函数，求解中心点与临近点之间的位置关系:
    ![LLE](/notebook/LLE_2.png)

    >ps.其中 $\underset{min}{arg}(w_i)$表示使等号后式子最大时得到 $w_i$的值， $X_i \in [1, D]$表示 $X_i的大小为1 \times D$,  $I_{k \times i}$为全1向量
    
    2.通过代数方法使函数最小，求得相对位置关系 $w_{i}$

    ![LLE](/notebook/LLE_3.png)

    ![LLE](/notebook/LLE_4.png)

    >ps.拉格朗日乘数法：设给定二元函数 $z=f(x,y)$ 和附加条件 $\varphi(x,y)=0$ 
为寻找 $z=f(x,y)$ 在附加条件下的极值点，先做拉格朗日函数
$F(x,y,\lambda)=f(x,y)+\lambda \varphi (x,y)$ ，其中 $\lambda$ 为参数。
令 $F(x,y,\lambda)$ 对 x 和 y 和 $\lambda$ 的一阶偏导数等于零，即
$F'_x=f'_x(x,y)+\lambda\varphi'_x(x,y)=0$ 
$F'_y=f'_y(x,y)+\lambda\varphi'_y(x,y)=0$ 
$F'_\lambda=\varphi (x,y)=0$ 
由上述方程组解出 x，y 及 $\lambda$ ，如此求得的 $(x,y)$ ，就是函数 $z=f(x,y)$ 在附加条件 
 $\varphi (x,y)=0$ 下的可能极值点。
若这样的点只有一个，由实际问题可直接确定此即所求的点。

    3.低维空间重映射
    ![LLE](/notebook/LLE_5.png)
    ![LLE](/notebook/LLE_6.png)
    >ps.对于为何可以用矩阵的迹代替原式子，我还没有答案，期望有同学帮忙解答。

    ![LLE](/notebook/LLE_7.png)
这样，我们得到了 $X_i$降维后的映射点 $Y_i$,LLE也就完成了！~~补线性代数去了~~

### The Proper Orthogonal Decomposition(POD)
This is an explaination for note.py, the idea is got from https://youtu.be/TcqBbtWTcIc?si=fw05Dq5k5amXK66s.
Our teacher has taught the lecture at https://youtu.be/SGOf8ST0nfo?si=8BP7n6Hbwdb_Tv7V.

Suppose we have data that is a fuction of both space and time.
$$y(x,t) = \sum_{m}^{j=1}u_{j}(x)a_{j}(t)$$

Obviously, we can not deal with it easily. If we wanna:
1. to understand the pertinent features present in the data;
2. to store/represent/manipulate the data;
3. to build a reduced-complexity model for the dynamics.

Now, we collect both spatial location and times sets:

$$ 
x: x_{1}, x_{2}, x_{3},..., x_{n}.\\
t: t_{1}, t_{2}, t_{3},..., t_{m}.
$$

We can assemble the data $y(x,t)$ into an $m$ x $n$ matrix:

$$
Y = \begin{bmatrix}
 y(x_{1},t_{1}) & y(x_{1},t_{2}) & \dots  & y(x_{1},t_{m})\\
 y(x_{1},t_{1}) & y(x_{1},t_{2}) & \dots & y(x_{2},t_{m})\\
\vdots  & \vdots  & \ddots & \vdots \\
 y(x_{n},t_{1}) & y(x_{n},t_{2}) & \dots & y(x_{n},t_{m})
\end{bmatrix}
$$

If we measure two components of velocity, $u$ and $v$, the matrix changes as follows:

$$
Y = \begin{bmatrix}
 u(x_{1},t_{1}) & u(x_{1},t_{2}) & \dots & u(x_{1},t_{m})\\
 u(x_{1},t_{1}) & u(x_{1},t_{2}) & \dots & u(x_{2},t_{m})\\
\vdots  & \vdots  & \ddots & \vdots \\
 u(x_{n},t_{1}) & v(x_{n},t_{2}) & \dots & v(x_{n},t_{m})\\
 v(x_{1},t_{1}) & v(x_{1},t_{2}) & \dots & v(x_{1},t_{m})\\
 v(x_{1},t_{1}) & v(x_{1},t_{2}) & \dots & v(x_{2},t_{m})\\
\vdots  & \vdots  & \ddots & \vdots \\
 v(x_{n},t_{1}) & v(x_{n},t_{2}) & \dots & v(x_{n},t_{m})
\end{bmatrix}
$$

If we can use a vector on the left which is only going to be a fuction of space and a vector on right which is only going to be a fuction of time, then the purpose that we wanna get may make sense. Just like the figure as follows.

![0](\notebook\n11.png)

This can be achieved through a Singular Value Decomposition(SVD):
$$Y = U \Sigma V^{*}$$

![1](\notebook\n12.png)
![2](\notebook\n13.png)
![3](\notebook\n14.png)

Through a code example, it will help us understand better.

```Python
import numpy as np
from numpy import linalg
import matplotlib as mpl
from matplotlib import pyplot as plt

#Define data
x = np.linspace(-2,2,401)
Nx = np.size(x)

amp1 = 1
x01 = 0.5
sigmay1 = 0.6

amp2 = 1.2
x02 = -0.5
sigmay2 = 0.3

dt = 0.01
Nt = 1001
tend = dt*(Nt-1)
t = np.linspace(0,tend,Nt) #time

omega1 = 1.3
omega2 = 4.1

y1 = amp1*np.exp(-((x-x01)**2)/((2*sigmay1**2)))
y2 = amp2*np.exp(-((x-x02)**2)/((2*sigmay2**2)))

Y = np.zeros([Nx,Nt],dtype='d')
for tt in range(Nt):
    Y[:,tt] = y1*np.sin(2*np.pi*omega1*t[tt]) + y2*np.sin(2*np.pi*omega2*t[tt])
U, S, VT = linalg.svd(Y, full_matrices= False)
```

Then we finish the data initialization, and show the data set as follows.

```Python
# show y1 and y2
plt.plot(x,y1,label='y1')
plt.plot(x,y2,label='y2')
plt.xlabel('x', fontsize=18)
plt.ylabel('y', fontsize=18)
plt.legend()
plt.show()
```

![4](\notebook\n15.png)

```Python
#plt all data
Tgrid, Ygrid = np.meshgrid(t,x)

#contour
plt.contour(Ygrid, Tgrid, np.abs(Y))
plt.xlabel('x', fontsize=18)
plt.ylabel('time', fontsize=18)
plt.ylim(0,4)
plt.show()
```
![5](\notebook\n16.png)
```Python
plt.semilogy(S,'-o')
plt.xlim(0,10)
plt.ylabel('Singular Value', fontsize=18)
plt.xlabel('Index', fontsize=18)
plt.show()
```
![6](\notebook\n17.png)
The diagnonal matrix

```Python
# x, U
plt.plot(x,U[:,0],label='U1')
plt.plot(x,U[:,1],label='U2')
plt.xlabel('x', fontsize=18)
plt.ylabel('U', fontsize=18)
plt.title('POD modes', fontsize=18)
plt.legend()
plt.show()
```
![7](\notebook\n18.png)

Each column of this matrix($U$) is a mode or a vector which is for our set of simulation snapshots can be unpack into a picture of the same size of original snapshots. Each one is a spatial mode constructed with the underlying data. The first mode is just the average of this data set. The first few modes should give us a pretty good approximation of the underlying data set.

![9](\notebook\n110.png)

```Python
# t,VT
plt.plot(t,VT[0,:],label='VT1')
plt.plot(t,VT[1,:],label='VT2')
plt.xlim(0,4)
plt.xlabel('time', fontsize=18)
plt.ylabel('VT', fontsize=18)
plt.title('mode coefficients', fontsize=18)
plt.legend()
plt.show()
```
![8](\notebook\n19.png)
