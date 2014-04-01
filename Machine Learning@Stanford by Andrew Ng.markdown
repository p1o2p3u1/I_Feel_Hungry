# Machine Learning@Stanford by Andrew Ng
##监督学习 Supervised Learning
在监督学习中，对于数据集中的每个数据，都有相应的正确答案（训练集），算法就是基于这些来做出判断。
### 回归问题 Regression Problem
预测一个连续值的输出，比如根据房屋大小输出房价的可能值
### 分类问题 Classification Problem
预测一个离散值的输出，如输出是或否，0或1，etc

##无监督学习 Unsupervised Learning
无监督学习中没有任何标签或者相同或者没有标签来表明数据的正误。对于已有的数据而不知如何去处理，也未被告知每个数据点是什么。对于这样的数据集，能从中找到某种结构吗？
###聚类算法
无监督学习能从一个数据集中找到不同的数据簇。如在谷歌新闻中，谷歌每天搜集大量的新闻内容，然后将这些新闻分组构成关联新闻，他们都是同一主题。

###鸡尾酒宴问题
有个宴会，里面有很多人，这么多人同时在聊天，声音彼此重叠。假设酒宴中有两个人，他俩同时在说话。我们放两个麦克风在他们面前，离说话人的距离不同，每个麦克风记录下不同的声音。这样虽然是同样的两个说话的人，可能离人近的麦克风的声音大一点，每个麦克风都听到的是两个人重叠的声音。我们要用鸡尾酒宴算法，找到声音数据里的结构，将混合的声音区分出来。

要实现这个算法，我们只需要一行代码
```
[W,s,v] = svd((repmat(sum(x.x,1),size(x,1),1).*x)*x');
```

Octave编程环境

SVD（支持向量机）

##学习模型
###监督学习——单参数线性回归（Linear regression with one variable）
Training set of housing prices
| 面积 | 价格/1000$ |
|------|:-------------|
|2104|460|
|1416|232|
|1534|315|
|852|178|

现在我们引入一些变量：

 - 我们用m来表示训练样本数据的大小
 - x表示输入变量/特征，y表示输出变量/目标变量
 - 我们用（x,y）表示一个训练样本数据，即表格里的一行。
 - 我们用$(x_i,y_i)$表示第i个样本，即表格里的第i行。

我们通过Training Set 得到Learning Algorithm，通过Learning Algorithm得到一个假设函数h（hypothesis），函数h的作用就是通过输入房屋的大小x，预测房屋的价格y。y=h(x)，或者
$$h_{\theta}(x) = {\theta}_0 + {\theta}_1 x$$

###Cost Function
现在我们已经有了假设函数
$$h_{\theta}(x) = {\theta}_0 + {\theta}_1 x$$
其中 ${\theta}_i$ 是该学习模型的参数。我们需要设定这两个参数的值，尼玛其实就是我们初中数学里的$f(x) = kx+b$

现在我们在平面坐标轴上已经有了很多样本点，现在我们需要画一条直线去尽可能多的逼近这些样本点，即找到函数里的k和b，并期望误差h(x) - y尽可能的小。同时我们又m个样本点，因此我们可以求期望值
$$E(x) = \frac 1 m \sum_{i=1}^m(h(x_i) - y_i)$$
而Cost Function与它类似
$$找到合适的\theta_0,\theta_1，使得J(\theta_0,\theta_1) = \frac 1 {2m} \sum_{i=1}^m(h(x_i) - y_i)^2 最小$$
Cost Function也称为平方误差函数，通过这个函数，我们容易找到慢速误差最小的h。
![pic](http://c.hiphotos.bdimg.com/album/s%3D550%3Bq%3D90%3Bc%3Dxiangce%2C100%2C100/sign=b1010162a964034f0bcdc2039ff8080c/faedab64034f78f00dcc554c7b310a55b2191c4e.jpg?referer=e5ca13abba99a90162226f065c45&x=.jpg)

### 梯度下降法 Gradient Descent
已知函数$J(\theta_0,\theta_1)$，想要得到$\min_{\theta_0,\theta_1}J(\theta_0,\theta_1)$。

**目标**

 - 从一些$\theta_0,\theta_1$出发
 - 改变$\theta_0,\theta_1$的值来得到$J(\theta_0,\theta_1)$，直到我们找到最小的J

**Gradient descent algorithm**
loop unitl convergence
{
$\theta_j:=\theta_j-\alpha\frac {d} {d\theta_j} J(\theta_0,\theta_1)$
}
其中$\alpha$表示learning rate，在更新$\theta_j$时控制我们一步要走多大。后面是一个偏微分
![pic2](http://f.hiphotos.bdimg.com/album/s%3D550%3Bq%3D90%3Bc%3Dxiangce%2C100%2C100/sign=6f03db6c1c178a82ca3c7fa5c63802b0/cefc1e178a82b901a7b5e293718da9773912ef72.jpg?referer=ba09054fabec8a134d0d62d04910&x=.jpg)
![pic3](http://g.hiphotos.bdimg.com/album/s%3D550%3Bq%3D90%3Bc%3Dxiangce%2C100%2C100/sign=9762955b3bdbb6fd215be523391fda25/80cb39dbb6fd526616ae1db4a918972bd507368c.jpg?referer=09440a878882b90164baf7037283&x=.jpg)

当得到一个最小$\theta_j$的时候，后面的微分（斜率）就为0，我们就得到了一个收敛的最优$\theta_j$。同时，尽管我们有一个固定的$\alpha$，但随着函数越来越接近最小值，后面的微分计算得到的值会越来越小，即梯度下降的范围会越来越小，因此我们无需每次就减小$\alpha$的值。
![pic4](http://d.hiphotos.bdimg.com/album/s%3D550%3Bq%3D90%3Bc%3Dxiangce%2C100%2C100/sign=7ccf3bef9c2f07085b052a05d91fc9a4/267f9e2f07082838e429f5b4ba99a9014c08f16e.jpg?referer=6b4b84b2700e0cf3f9e07bcb3f24&x=.jpg)

###Gradient descent for linear regression
现在，我们有了Gradient descent algorithm和Linear Regression Model（linear hypothesis和平方误差函数）。现在我们需要用学习算法Gradient descent algorithm，对模型Linear Regression Model求得最优假设函数。
$$j=0:\frac d {d\theta_0}J(\theta_0,\theta_1) = \frac 1 m \sum_{i=1}^m(h_{\theta}(x_i) - y_i) (求偏导数)$$

$$j=1:\frac d {d\theta_1}J(\theta_0,\theta_1) = \frac 1 m \sum_{i=1}^m(h_{\theta}(x_i) - y_i)*x_i$$

因此利用上面的得到的公式，使用**Gradient descent algorithm**

repeat until convergence
{

$\theta_0 := \theta_0 - \alpha \frac 1 m \sum_{i=1}^m (h_{\theta}(x_i) - y_i )$

$\theta_1 := \theta_1 - \alpha \frac 1 m \sum_{i=1}^m (h_{\theta}(x_i) - y_i )*x_i$

}
Note: Each step of gradient descent uses all the training examples.

## 线性代数复习（Linear Algebra Review）SKIP

##多个变量下的线性回归（Linear Regression with Multiple Variables）
###多变量的引入与向量表示
我们回到预测房屋价格的那个模型，现在引入多个变量
|房屋大小$x_1$|房间数$x_2$|层数$x_3$|房屋年龄$x_4$|价格$y$|
|---------|:------|:----|:-----|:-----|
|2104|5|1|45|460|
|1416|3|2|40|232|
|1534|3|2|30|315|
|852|2|1|36|178|

现在我们引入一些变量：

 - n表示变两个数（表格里的x的个数）
 - $x_i$表示第i个样本
 - $x_i^j$表示第i个样本的第j个变量

我们的假设函数h（hypothesis）是一个n元一次方程
$$h_\theta(x) = \theta_0+\theta_1x_1+\theta_2x_2+……+\theta_nx_n$$
我们可以假设$x_0=1$，这样我们可以将变量以向量的形式表示：
$$X=
\left( {\begin{array}{*{20}{c}}
   {x_0}   \\
   {x_1}   \\
   {x_2}   \\
   \vdots  \\
   {x_n}   \\
\end{array}} \right) 
\in R^{n+1},\Theta=
\left( {\begin{array}{*{20}{c}}
   {\theta_0}   \\
   {\theta_1}   \\
   {\theta_2}   \\
   \vdots  \\
   {\theta_n}   \\
\end{array}} \right) 
\in R^{n+1}
$$
因此我们的假设函数可以表示为
$$h_\theta(x) = \Theta^TX$$
###在多变量下使用梯度下降法（Gradient Descent for Multiple Variables）
Cost function 表示为
$$J(\theta_1,\theta_2,...,\theta_n) = \frac 1 {2m} \sum_{i=1}^n(h_\theta(x_i)-y_i)^2$$

Gradient descent :
Repeat
{
$\theta_j := \theta_j - \alpha \frac d {d\theta_j}J(\theta_0,...,\theta_n)$  
对每个$j=0,...,n$都要同时更新。
}

将Cost function带入进去对每一个$\theta_j$求偏微分，得到

Repeat
{
$\theta_j := \theta_j - \alpha \frac 1 m \sum_{i=1}^m(h_\theta(x_i) - y_i)*x_i^j$  
对每个$j=0,...,n$都要同时更新。
}

###Gradient Descent in Practice : Feature scaling
使得每个变量的取值范围都大概处以同一范围，这样得到的图像更像一个圆形，使得得到minJ的速度更快。如

$x_i := \frac {x_i-\mu} {S}$，其中$\mu$表示训练集x的均值，$S$表示样本x里的（max-min）的值。

###Gradient Descent in Practice：Learning Rate

 - 如何确保gradient descent是在正确的工作？
 - 如何选择一个合适的Learning rate $\alpha$ ？

正确的Gradient descent工作模型
![pic5](http://g.hiphotos.bdimg.com/album/s%3D550%3Bq%3D90%3Bc%3Dxiangce%2C100%2C100/sign=b929bd7e74c6a7efbd26a823cdc1de6c/91ef76c6a7efce1bc32ed06bad51f3deb48f650d.jpg?referer=f8268542ff1f4134b920304ec303&x=.jpg)

错误的Gradient descent工作模型
![pic6](http://g.hiphotos.bdimg.com/album/s%3D550%3Bq%3D90%3Bc%3Dxiangce%2C100%2C100/sign=12cdaad5e9f81a4c2232eccce7111164/8644ebf81a4c510fc707fabb6259252dd52aa54b.jpg?referer=120a1c3cc9ef7609651cadaf0349&x=.jpg)

选择合适的Learning Rate $\alpha$，尝试下列取值  
0.001， 0.003， 0.01， 0.03， 0.1， 0.3， 1，…… 

###Features and Polynomial Regression
线性回归弱爆了
![pic7](http://f.hiphotos.bdimg.com/album/s%3D550%3Bq%3D90%3Bc%3Dxiangce%2C100%2C100/sign=f6b14e984936acaf5de096f94ce2fc2d/77094b36acaf2edd9e9244488f1001e93901932b.jpg?referer=613e0c395cdf8db1e5394854dde9&x=.jpg)

###（标准方程）Normal Equation
我们再次回到使用多个变量预测房屋价格的那个模型
|$x_0$|房屋大小$x_1$|房间数$x_2$|层数$x_3$|房屋年龄$x_4$|价格$y$|
|----|:---------|:------|:----|:-----|:-----|
|1|2104|5|1|45|460|
|1|1416|3|2|40|232|
|1|1534|3|2|30|315|
|1|852|2|1|36|178|
我们设置$x_0$都为1，我们可以用矩阵来表示这些变量的值
$$
X=
\left[ {\begin{array}{*{20}{c}}
1&2104&5&1&45 \\
1&1416&3&2&40 \\
1&1534&3&2&30 \\
1&852&2&1&36 \\
\end{array}} \right] 
y=
\left[ {\begin{array}{*{20}{c}}
460 \\
232 \\
315 \\
178 \\
\end{array}} \right] 
$$
$X$是一个m行n+1列矩阵，$y$是一个m维列向量。则$\theta$可由如下等式表示
$$\theta = (X^TX)^{-1}X^Ty$$
在Octave中可用如下代码表示
```
pinv(X'*X)*X'*y
```
比较一下Gradient Descent和Normal Equation
| m个样本n个变量   |  Gradient Descent  | Normal Equation  |
|---|:---------------------|:--------------------|
|$\alpha$|需要选择$\alpha$|不需要选择$\alpha$|
|iteration|需要多轮iterations|不需要iterate|
|复杂度|当n比较大时效率也很高|需要进行矩阵运算$O(n^3)$，当n比较大时用时较长|

###不用求逆的标准方程(Normal Equation Non-invertibility)
已知Normal Equation如下表示
$$\theta = (X^TX)^{-1}X^Ty$$
倘若$(X^TX)$不可逆的话怎么办呢？（尽管出现的概率很小）

 - 有一些冗余的变量之间存在线性相关，删之
 - 有太多的变量，使得（$m \leq n$），删之，或者正则化(regularization)

##学习使用Octave编程环境
```matlab
Basic：
1 ~= 2     % 不等号用~=表示
PS1('>> ')    % 改变控制台的输入符号
disp(sprintf('2 decimals : %0.2f',a))  % 高级输出方式
A=[1 2; 3 4; 5 6]   % 定义一个三行两列矩阵
B = [1 2 3]  % 定义一个行向量
C = [1;2;3]   % 定义一个列向量
v = 1:0.1:2  % 定义一个行向量，从1开始，每次增加0.1，直到2
ones(2,3)  % 定义一个2*3的矩阵，里面的元素都是1，此外还有zeros(m,n)
2*ones(2,3) %定义一个2*3的矩阵，里面的元素都是2
rand(2,3)  %定义一个2*3的矩阵，里面的元素都是从0-1的随机数，randn(m,n) 产生负的随机数
w = -6 + sqrt(10) * randn(1,10000)
hist(w) % 输出w的直方图
eye(4)  %输出一个4*4的单位矩阵

Move data
size(A) %输出A是一个几行几列的矩阵，是一个行向量
length(A) %输出A的维度。若A是一个矩阵，则输出最大的维度
cd 'path'  % 改变当前环境的位置
load ex1data1.txt  % 载入当前环境下的ex1.data1.txt里的文件
who % 显示当前已有的变量
whos % 显示当前变量详情
save hello.mat v; % 将变量v的值保存到文件hello.mat中（matlab文件）
save hello.txt v -ascii ; % 将变量v的值以文本文件形式保存
A = [1 2;3 4;5 6]
A =
   1   2
   3   4
   5   6

A(3,2) % 得到6，即第三行第二列
A(2, :) % 得到第二行所有列，即得到一个行向量
A(:, 2) % 得到第二列所有行，即得到一个列向量
A([1 3], :) % 得到第一行，第三行的所有列
A(:, 2) = [8; 9; 10]  % 将A的第二列的所有行重新赋值
A = [A, [10;11;12]]; % 给A添加一列，相当于A = [A,B] 或 [A B] 两个矩阵合并
A = [A; B] % 将两个矩阵合并，只不过是向列的方向合并

Computing on Data
A*B % 矩阵乘法
A .*B % 算数乘法，即A中的每个元素，与对应的B中的元素相乘，又比如A .^ 2 即A中的每个元素都平方
log(C) %对C中的每个元素求log，其他如exp(C), abs(C), -C,
C' % C转置'
[val,ind] = max(C) % 将矩阵C的每个列向量的最大值赋值给val，该最大值的所在行赋值给ind
max(C,[],1) % 得到矩阵C所有列向量的最大值，相应的max(C,[],2) 得到C所有行向量的最大值
max(C(:)) % 得到矩阵C所有元素的最大值
pinv(C) % 对矩阵C求逆

Ploting Data
t = [0:0.01:1];
y1=sin(2*pi*4*t);
plot(t,y1);
hold on;
y2 = cos(2*pi*4*t);
plot(t,y2,'r');
xlabel('time');
ylabel('value');
legend('sin','cos');
title('my plot');
print -dpng 'myplot.png'

figure(1) ;plot(t,y1);
figure(2);plot(t,y2);

subplot(1,2,1); % 将plot分成1*2的表格，将接下来的plot放到第一个表格里。
plot(t,y1);
subplot(1,2,2);
plot(t,y2);
axis([0.5 1 -1 1]) %改变x，y坐标轴的取值范围

Control Statement
for i=1:10,
    do something;
end;

while i <=5,
    do something;
end;

if true,
    do something;
end;

function y = compute(x)  % 不添加y，控制台就没有输出
x = x + 1; 
end;

function [y1,y2] = compute(x)
y1 = x + 1;
y2 = x .^ 2;
end;

[a,b] = compute(5);
```
Vectorization
![pic8](http://f.hiphotos.bdimg.com/album/s%3D550%3Bq%3D90%3Bc%3Dxiangce%2C100%2C100/sign=8761a280b27eca80160539e2a118e6e0/9213b07eca80653839a7a0b095dda144ac3482c7.jpg?referer=929fff889f16fdfa817bf2deeedd&x=.jpg)

## 逻辑回归（Logistic Regression）
### 分类问题 Classification
垃圾邮件分类  
在线交易：是否是欺诈交易  
肿瘤：恶性，良性。
Let's start with binary classification problem 
$$y \in \{ 0,1 \}$$
线性规约在此处不适用，我们需要使用Logistic Regression：$0 \leq h_\theta(x) \leq 1$。尽管它有个Regression的名字，但做的还是Classification的事情我去！
###Hypothesis Representation
我们想要假设函数的取值范围在0~1之间，即$0 \leq h_\theta(x) \leq 1$。我们定义两个函数

Sigmoid function: $g(z) = \frac 1 {1 + e^{-z}}$

Logistic function: 令$z = h_\theta(x) = \Theta^Tx$，带入到sigmoid function中。
![pic9](http://a.hiphotos.bdimg.com/album/s%3D550%3Bq%3D90%3Bc%3Dxiangce%2C100%2C100/sign=e896da1467380cd7e21ea2e8917fdc09/cb8065380cd79123cf35b1d0af345982b3b780b8.jpg?referer=c5feed6ead51f3de9aa58d54e056&x=.jpg)

### Decision boundary
我们已经有了Logistic regression的表达式$h_\theta(x) = g(\Theta^Tx)，g(z) = \frac 1 {1+e^{-z}}$，其图像如下所示
![pic10](http://c.hiphotos.bdimg.com/album/s%3D550%3Bq%3D90%3Bc%3Dxiangce%2C100%2C100/sign=c0a0898bbc096b6385195e553c08f679/f31fbe096b63f624a18677a78544ebf81b4ca3bf.jpg?referer=5d3bdad0af3459829c9dd1a20d55&x=.jpg)
给定一定的训练集，我们可以得到Decision boundary如下所示
![pic11](http://a.hiphotos.bdimg.com/album/s%3D550%3Bq%3D90%3Bc%3Dxiangce%2C100%2C100/sign=f1254a68c1cec3fd8f3ea770e6b3a502/c9fcc3cec3fdfc03da72541cd63f8794a4c2260f.jpg?referer=1298269b49fbfbed854e034f8205&x=.jpg)

### Cost Function
我们有m个样本的训练集合$\{ (x_1,y_1),(x_2,y_2),…,(x_m,y_m) \}$，我们有n+1个特征$x \in \left[ {\begin{array}{*{20}{c}}
   {x_0}   \\
   {x_1}   \\
   {x_2}   \\
   \vdots  \\
   {x_n}   \\
\end{array}} \right] $，其中$x_0 = 1,y \in \{ 0,1 \}，h_\theta(x) = \frac 1 {1 + e^{-\theta^Tx}}$。如何选择参数$\theta$？

根据我们以前的Linear Regression的公式，
$$J(\theta) = \frac 1 m \sum_{i=1}^m \frac 1 2 (h_\theta(x_i) - y_i )^2$$
我们令
$$Cost(h_\theta(x),y) = \frac 1 2 (h_\theta(x) - y )^2$$
现在，因为我们的$h_\theta(x)$不再是一个Linear function，如果还是用上面的公式的话，我们最后得到的关于$J(\theta)$的函数就不再是一个凸函数，使得我们难以得到最优解。
![pic12](http://g.hiphotos.bdimg.com/album/s%3D550%3Bq%3D90%3Bc%3Dxiangce%2C100%2C100/sign=2f2f216d808ba61edbeec82a710fe637/32fa828ba61ea8d3630553a3950a304e251f5875.jpg?referer=3374cf5353da81cb17f1b6fd982b&x=.jpg)
因此我们需要定义一个新的Cost function，使得对于logistic regression得到的$J(\theta)$函数依然是凸函数。

**Logistic regression cost function**
$$Cost(h_\theta(x),y) =  \begin{equation}
  \left\{
   \begin{aligned}
   \overset{.} -log(h_\theta(x)) \quad  if\ y = 1\\
   -log(1-h_\theta(x)) \quad if\ y=0\\
   \end{aligned}
   \right.
  \end{equation} $$
![pic13](http://g.hiphotos.bdimg.com/album/s%3D550%3Bq%3D90%3Bc%3Dxiangce%2C100%2C100/sign=96c7bdb2be315c6047956beabd8aba2e/9825bc315c6034a828e6a926c913495409237614.jpg?referer=4dfebbaafadcd100948bcd11b20a&x=.jpg)
![pic14](http://e.hiphotos.bdimg.com/album/s%3D550%3Bq%3D90%3Bc%3Dxiangce%2C100%2C100/sign=094ae3cb1238534388cf8724a328c143/c75c10385343fbf28e71b684b27eca8065388f33.jpg?referer=b955c04ae51190ef58eca6efe9d1&x=.jpg)

###简化的cost function和gradient descent
我们已经有cost function
$$J(\theta) = \frac 1 m \sum_{i=1}^m Cost(h_\theta(x_i),y_i)$$
$$Cost(h_\theta(x),y) =  \begin{equation}
  \left\{
   \begin{aligned}
   \overset{.} -log(h_\theta(x)) \quad  if\ y = 1\\
   -log(1-h_\theta(x)) \quad if\ y=0\\
   \end{aligned}
   \right.
  \end{equation} $$
注意到我们的Cost function是分两种情况的，我们想将这两个方程压缩成一个方程，方便编程实现和Gradient descent。
$$Cost(h_\theta(x),y) = -ylog(h_\theta(x)) - (1-y)log(1-h_\theta(x))$$
$$J(\theta) = -\frac 1 m [\sum_{i=1}^m y_i log(h_\theta(x_i)) + (1-y_i)log(1-h_\theta(x_i)) ]$$
为了得到合适的$\theta$：$min_\theta J(\theta)$
对于给定的样本$x$：输出$h_\theta = \frac 1 {1 + e^{-\theta^Tx}}$
为了得到$min_\theta J(\theta)$：Gradient descent
Repeat
{
$\theta_j := \theta_j - \alpha \frac d {d\theta_j} J(\theta)$
对于每个$\theta_j$都要同时更新。
}
带入$J(\theta)$求偏微分，我们会发现得到的等式跟我们以前的Linear regression是一样的
$$\theta_j := \theta_j - \alpha \sum_{i=1}^m(h_\theta(x_i) - y_i)x_i^j$$
只是$h_\theta(x)$的定义不一样了。

###高级优化
我们有一些其他的优化算法：

 - Conjugate gradient
 - BFGS
 - L-BFGS

它们都有一些共同的特点：

 - 无需选择learning rate $\alpha$
 - 速度会比gradient descnet快
 - 但是比较复杂。因此最好的方法是使用已经写好的库，而不要自己去造轮子。

```
function [jVal,gradient] = costFunction(theta),
    jVal = (theta(1) - 5)^2 + (theta(2) - 5)^2;
    gradient = zeros(2,1);
    gradient(1) = 2*(theta(1) - 5);
    gradient(2) = 2*(theta(2) - 5);
end;

options = optimset('GradObj','on','MaxIter','100');
initialTheta = zeros(2,1);
[optTheta,functionVal,exitFlag] = fminunc(@costFunction,initialTheta,options);

返回
optTheta =
   5
   5
functionVal = 0
exitFlag =  1
```

###多种类型分类问题
比如我想将收到的邮件分成多类：工作，朋友，家庭，兴趣。对于给定的样本，如何将这些样本数据按照要求进行分类。

我们可以用Logistic regression进行二维分类（是或否，0或1），我们使用one-vs-all来解决这个问题
![one-vs-all](http://b.hiphotos.bdimg.com/album/s%3D550%3Bq%3D90%3Bc%3Dxiangce%2C100%2C100/sign=de358985cafcc3ceb0c0c936a27ea7b5/b812c8fcc3cec3fd6526a2d2d488d43f87942778.jpg?referer=1b276a38a60f4bfbd5c7ab648116&x=.jpg)
即对每种分类构造分类函数$h_\theta^i(x)$，并用Logistic regression进行求解。

##Regularization
###过适问题（problem of overfitting）
![overfitting](http://b.hiphotos.bdimg.com/album/s%3D550%3Bq%3D90%3Bc%3Dxiangce%2C100%2C100/sign=67b41db9bb389b503cffe057b50e94e0/2e2eb9389b504fc28407b7a2e7dde71190ef6d38.jpg?referer=9dda5e16820a19d89214b035cbd6&x=.jpg)
![overfitting2](http://h.hiphotos.bdimg.com/album/s%3D550%3Bq%3D90%3Bc%3Dxiangce%2C100%2C100/sign=b8fe7030db33c895a27e987ee12802cd/43a7d933c895d1431a91e38471f082025baf07b5.jpg?referer=fd371ce489d4b31ca92ba08b616c&x=.jpg)
即对于我们的假设函数h，overfitting使得我们的假设函数对于已知的数据预测非常准确，但对于未知的，新的数据却不能得到准确的预测（即对于Training set过度适应）。

如何检测出underfitting和overfitting？可选的做法有：

 - 减少feature的数量，包括人工选择和通过算法选择。缺点是减少了feature，就相当于丢失了问题的一部分信息。
 - Regularization。保留所有的feature，但对每个feature的权值降低（较小的theta值），使得每个feature对预测y都贡献一点。

###Cost function
假设我们的假设函数$h = \theta_0 + \theta_1x + \theta_2x^2 + \theta_3x^3 + \theta_4x^4$。我们的假设函数overfit，使得$\theta_3,\theta_4$非常小，$\theta_3 \approx 0,\theta_4 \approx 0$。这样$x^3，x^4$所起到的作用就变得很小。

**Regularization**

 - 对于参数theta给予较小的权值，使得我们的假设函数得以简化，降低overfit的可能性。
 - 以前面预测房价的那个为例。假设我们有100个feature，我们不知道哪个feature会有较高的幂，因此我想修改cost function
$$J(\theta) = \frac 1 {2m} \sum_{i=1}^m[(h_\theta(x_i) - y_i)^2 + \lambda \sum_{i=1}^m \theta_i^2]$$
后面新出现的部分称之为Regularization，参数$\lambda$称为Regularization参数，它的作用是用来平衡前后两个部分，使得前半部分能够适应training set，后半部分能够让我们得到较小的theta参数，最终能让我们的假设函数简化以防止overfitting。
![underfitting](http://g.hiphotos.bdimg.com/album/s%3D550%3Bq%3D90%3Bc%3Dxiangce%2C100%2C100/sign=166a081867380cd7e21ea2e8917fdc09/cb8065380cd7912331c963dcaf345982b2b78069.jpg?referer=3b023f62ad51f3de9aa58c54e027&x=.jpg)

### Regularized Linear Regression