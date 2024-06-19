# 傅里叶级数

周期函数可以被不同频率和相位三角函数的和表示

$$
\begin{align}
F_l(x)\sim S(x)
&= \frac{a_0}{2}+\sum_{n=0}^\infty\left( a_n\cos \frac{n\pi x}{l}+  b_n\sin \frac{n\pi x}{l} \right)
\\
&= \sum_{n=0}^\infty\left( a_n\cos \frac{n\pi x}{l}+  b_n\sin \frac{n\pi x}{l} \right)
\\
&= \sum_{n=0}^\infty c_n\left(\exp{in\frac{2\pi t}{l}}\right)
\\
a_n 
&= \frac{1}{l}\int_{-l}^l F_l(t)\cos\frac{n\pi t}{l}{\rm d}t
\\
b_n 
&= \frac{1}{l}\int_{-l}^l F_l(t)\sin\frac{n\pi t}{l}{\rm d}t

\\
c_n 
&= \frac{2}{l}\int_{-l}^l F_l(t)\left(\exp{in\frac{n\pi t}{l}}\right){\rm d}t
\end{align}

$$

# 傅里叶变换

非周期函数看作周期无穷大的周期函数