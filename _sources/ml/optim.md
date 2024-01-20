# Optimization, GradientDescent (최적화와 경사하강법)

- 이 글은 2019년도에 Andrew ug 교수님의 coursera강의를 듣고 쓴 글임을 밝힙니다.

본 포스팅은 Beale Function을 이용하여 한번 다양한 NN optimzation 함수를 구현해보고 해결해 보려고 합니다! 각 최적화 기법들의 특징들을 살펴봅시다. 일반적인 Gradient Descent의 문제점은, Local Minimum 과 Saddle point에 빠지는 경우의 수입니다.

## Gradient Descent
Gradient descent는 제약조건이 없는 convex이고 differentiable한 function의 최적화 문제를 풀기위한 가장 단순한 알고리즘입니다.    

$$
w_{t+1} = w_{t} - \eta_{t} \frac{\partial \mathcal{L}}{\partial w_{t}}
$$

## Stochastic Gradient Descent
전체 데이터에서 구하지 않고 mini-batch로 랜덤하게 샘플링하여 loss를 구하게 됩니다.

## Momentum method 
파라미터들을 단지 현재의 gradient만 고려하지말고 history 역시 고려해보자는게 주요한 아이디어입니다.
이것은 마치 leaky integrator (IIR filter)와 비슷한 기능을 하게 된다. Gradient에 관성을 넣어주는 것이죠.

- Velocity variable : v를 도입해봅시다.
- direction,speed(속력)은 파라미터의 space에서 파라미터의 움직임을 따르게됩니다.
- 모멘텀은 물리학에서 mass(질량) x velocity(속도) 입니다.
- unit 을 mass로 생각해볼 수 있습니다.
- hyperparameter γ∈[0,1) determines exponential decay

$$
v_{t+1} = \gamma v_{t} + \frac{\partial \mathcal{L}(w_{t})}{\partial w_{t}}
$$

$$
w_{t+1} = w_{t} - \eta v_{t+1}
$$

- $v_{t}$ : 누적된 gradients의 관성
- $w_{t}$ : Gradients (가속도)


```python
class MomentumOptimizer():
    def __init__(self, function, gradients, learning_rate=0.01, momentum=0.9):
        self.f = function
        self.g = gradients
        scale = 3.0
        self.vars = np.zeros([2])
        self.vars[0] = np.random.uniform(low=-scale, high=scale)
        self.vars[1] = np.random.uniform(low=-scale, high=scale)
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = np.zeros([2])
        
        # for accumulation of loss and path (w, b)
        self.z_history = []
        self.x_history = []
        self.y_history = []
  
    def func(self, variables):
        x, y = variables
        z = self.f(x, y)
        return z
    
    def gradients(self, variables):
        x, y = variables
        grads = self.g(x, y)
        return grads
    
    def weights_update(self, grads):
        """
          v' = gamma * v + dL/dw
          w' = w - lr * v'
        """
        self.velocity = self.momentum * self.velocity + grads
        self.vars = self.vars - self.lr * self.velocity
        
    
    def weights_update1(self, grads):
        """
          Weights update using Momentum.

          v' = gamma * v - lr * dL/dw
          w' = w + v'
        """
        self.velocity = self.momentum * self.velocity - self.lr * grads
        self.vars = self.vars + self.velocity
    
    def history_update(self, z, x, y):
        self.z_history.append(z)
        self.x_history.append(x)
        self.y_history.append(y)

    def train(self, max_steps):
        pre_z = 0.0
        print("steps: {}  z: {:.6f}  x: {:.5f}  y: {:.5f}".format(0, self.func(self.vars), self.x, self.y))

        for step in range(max_steps):
            self.z = self.func(self.vars)
            self.history_update(self.z, self.x, self.y)
            self.grads = self.gradients(self.vars)
            self.weights_update1(self.grads)

            if (step+1) % 100 == 0:
                print("steps: {}  z: {:.6f}  x: {:.5f}  y: {:.5f}  dx: {:.5f}  dy: {:.5f}".format(step+1, self.func(self.vars), self.x, self.y, self.dx, self.dy))

            if np.abs(pre_z - self.z) < 1e-7:
                print("Enough convergence")
                print("steps: {}  z: {:.6f}  x: {:.5f}  y: {:.5f}".format(step+1, self.func(self.vars), self.x, self.y))
                self.z = self.func(self.vars)
                self.history_update(self.z, self.x, self.y)
                break

            pre_z = self.z

        self.x_history = np.array(self.x_history)
        self.y_history = np.array(self.y_history)
        self.path = np.concatenate((np.expand_dims(self.x_history, 1), np.expand_dims(self.y_history, 1)), axis=1).T

    @property
    def x(self):
        return self.vars[0]

    @property
    def y(self):
        return self.vars[1]

    @property
    def dx(self):
        return self.grads[0]

    @property
    def dy(self):
        return self.grads[1]
```


```python
Momentum = MomentumOptimizer(f, gradients, x_init=0.7, y_init=1.4, learning_rate=0.01, momentum=0.9)
```

    x_init: 0.700
    y_init: 1.400



```python
Momentum.train(1000)
```

    steps: 0  z: 26.496662  x: 0.70000  y: 1.40000
    steps: 100  z: 0.099510  x: 4.46410  y: 0.71417  dx: 0.09680  dy: -0.29558
    steps: 200  z: 0.039972  x: 3.70210  y: 0.63209  dx: 0.07884  dy: 0.01149
    steps: 300  z: 0.001087  x: 3.08637  y: 0.52077  dx: 0.02342  dy: 0.00558
    Enough convergence
    steps: 379  z: 0.000001  x: 3.00194  y: 0.50048



```python
print("x: {:.4f}  y: {:.4f}".format(Momentum.x, Momentum.y))
```

    x: 3.0019  y: 0.5005


## Adagrad

* 경사에 따라서 learning rate를 조금 다르게 해보는 접근. updates를 조절합니다. 
* 파라미터마다 다른 learning rate를 주자 (Adaptive Learning rate).
* 큰 Variation (Gradient) 전에 많이 움직임 : 자주 업데이트 되는 파라미터
* 작은 Variation (Gradient) 전에 많이 안 움직임 : 자주 업데이트 안되는 파라미터
* Learning rate를 Gradient Variation로 나눠서, 수식을 바꾼것이다.
    * Problem : Decays to zero -> 학습속도가 매우 느려진다.

## RMSprop

* 학습 속도의 감소를 막아주는 효과가 있습니다.
* Moving average of squared gradient : 무지하게 커지니까 앞에 있는걸, Learning rate의 패널티를 moving average 해주게 됩니다.
* 학습시 Gradient의 미분식에 Normalizae panelity가 들어가는 것입니다.

## Adam
Adam은 gradient의 1차 모먼트 $m_{t}$ 와 2차 모먼트 $v_{t}$ 를 통하여, 모멘텀 효과와 weight마다 다른 learning rate를 적용하는 adaptive learning rate 효과를 지원하는 최적화 알고리즘입니다.

* Adaptive moment estimation입니다.
* RMSprop + Momentum 입니다.
* moving average of past and past squared gradients

$$
\begin{matrix} m_t & = & \beta_1 m_{t-1} + (1-\beta_1)\nabla f_t(w_t) \\\ 
v_t & = & \beta_2 v_{t-1} + (1-\beta_2)\nabla f_t(w_t)^2 \\\ 
\hat{m_t} & = & \frac{m_t}{1-\beta_1^t} \;\;\; \text{(Bias correction)}\\\ 
\hat{v_t} & = & \frac{v_t}{1-\beta_2^t} \;\;\; \text{(Bias correction)}\\\ 
w_{t+1} & = & w_{t} - \frac{\alpha}{\sqrt{\hat{v_t}} + \epsilon}\hat{m_t} \end{matrix}
$$

수식 출처는 [hiddenbeginner 블로그](https://hiddenbeginner.github.io/deeplearning/paperreview/2019/12/29/paper_review_AdamW.html) 입니다.


## AdamW

**AdamW**는 Adam 옵티마이저의 변형 중 하나로, 가중치 감쇠(Weight Decay)를 효과적으로 처리하기 위해 고안되었습니다. weight decay는 gradient descent에서 weight 업데이트를 할 때, 이전 weight의 크기를 일정 비율 감소시켜줌으로써 오버피팅을 방지하는 기법입니다. AdamW는 기존 Adam에 가중치 감쇠 항을 더함으로써 가중치 감쇠를 더 효과적으로 조절합니다. 따라서 AdamW는 가중치 감쇠에 대한 감도를 높이고 모델의 일반화 성능을 향상시킬 수 있습니다. AdamW는 다음과 같이 가중치 감쇠를 적용한 갱신 규칙을 사용합니다.

$$
w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \odot (\hat{m}_t + \lambda w_t)
$$

여기서 $w_t$는 가중치, $\eta$는 학습률, $\hat{m}_t$와 $\hat{v}_t$는 이동 평균으로서 Adam의 구성 요소입니다. 또한, $\lambda$는 가중치 감쇠 계수를 나타냅니다.
