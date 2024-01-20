# Feed forward Network (순방향 신경망)

- 이 글은 2019년도에 Andrew ug 교수님의 coursera강의를 듣고 쓴 글임을 밝힙니다.

이 포스팅에서는 backpropagation을 Feedforward Network를 활용해서 구현해 봅니다. Deep feedforward network, 흔히 말하는 Multi-layer perceptron은 어떤 function $f^{*}$에 근사하는 함수를 찾아가는 모형입니다. Classifier는 $y=f^{*}(x)$에서 $f^{*}$를 뜻하게 됩니다. 이는 input $x$와 category $y$를 mapping 하는 함수라는 것입니다. 우리의 feed forward network는 $y=f(x;\theta)$의 mapping 관계를 정의합니다. 이때 함수의 파라미터인 $\theta$를 학습하며, 최적의 function approximation를 찾아나가게 됩니다. function $f^{*}$는 우리의 이상적인 분류 모델이라고 생각하면됩니다.

## Feed forward

feedforward라고 부르는 이유는, 학습의 정보의 흐름을 보시게 된다면 데이터 $x$로 부터 시작되서, 함수 $f$를 정의하고, 정의된 함수를 통해 output 인 $\hat{y}$를 산출하게 됩니다. 이 과정에서 $\text{feedback connection}$이 부제하게 됩니다. 이 $\text{feedback connection}$은 output이 모델에 스스로 feedback을 보내는 연결관계입니다. backpropagation과는 다른 의미입니다. 이는 후에 $\text{Recurrent Neural Net}$ 에서 구현이 됩니다.

## Network

이제 Network의 의미를 살펴봅시다. 일반적으로는 differnet function이 함께 모델을 표현하기 때문에 Network라는 표현이 사용되었습니다. 이 모델은 일반적으로 directed acyclic graph로 표현이 됩니다. 예를 들면,

$$
f(x) =f^{3}(f^{2}(f^{1}(x)))
$$

이러한 체인 구조가 될 것입니다. 이 케이스에서 $f^{1}$은 네트워크의 첫번째 레이어, $f^{2}$는 두번쨰 레이어가 될 것으로 예상됩니다. 이러한 함수들이 depth를 가지고 존재하기 때문에 deep이라는 이름이 붙여지게 됩니다. 그리고 $f^{3}$인 마지막 레이어는 흔히 output layer 라고 불리게 됩니다. 뉴럴넷을 학습시키면서 우리는 이 $f(x)$를 $f^{*}(x)$에 근사시키고자 노력하게 됩니다. 

## Neural

이제 Neural의 의미를 살펴봅시다. 일반적으로는 우리는 neuroscience로 부터 이 용어를 쉽게 볼 수 있습니다. 각각 뉴럴넷의 hidden layer는 vector의 형태로 표현이 됩니다. 이 vector는 뉴런의 역할로 해석이 가능합니다. 그리고 각 Layer들은 vector-to-vector function으로 표현이 가능합니다. 그리고 Layer들 안에 있는 각각의 노드 즉 unit들은 vector-to-scalar function으로 생각할 수 있습니다. 각 unit들은 계산과 activation의 역할을 하게 됩니다. 이는 뇌의 뉴런들의 기능과 매우 유사합니다. 그렇다고 완벽한 뇌의 모델과 동일한 것은 아닙니다. 

## Universal Approximation Theorem

Feedforward netwrok가 linear ouput layer와 적어도 하나의 충분한 unit을 가지는 hidden layer를 가지게 된다면, Borel measurable function에 근사하게 됩니다. layer를 충분히 많이 사용하면, 어떠한 형태의 함수와도 유사한 형태의 함수 $f^{*}(x)$를 만들 수 있다고 합니다. 하지만 이 theorem은 얼마나 큰 network인지는 알려주지 않습니다. 

위 Theorem을 요약하자면 적어도 하나의 hidden layer를 가진 뉴럴넷은 any function을 표현할수 있지만, infeasibly large해야하기 때문에, generalize correctly한 모델을 만드는데 실패합니다. 때문에 Deeper models을 사용을 해서, generalize error를 줄여나가야 합니다.

## Multi-Layer-Perceptron

```{figure} ../images/classification.png
---
name: classification
---
Multi-Layer-Perceptron
```

$$
\text{Output layer} = \sigma \left( \sum_{j=1}^M w  \text{tanh} \left(w_{j}^{(1)} x + b_j^{(1)} \right)  + b \right) \\
\text{Hidden layer} = \text{tanh} \left(w_{j}^{(1)} x + b_j^{(1)} \right)
$$

$$
\sigma(x) = \frac{1}{1+e^{-x}}
$$

##  Linear & Non-linear
위 간단한 뉴럴넷의 모양을 살펴보면 2가지 구간으로 나눌수 있습니다. 첫번쨰는 $w_{j}^{(1)} x + b_j^{(1)}$ 로 표현되는 Affine transform의 구간입니다. 그리고 이 결과는 $\text{tanh}$ 혹은 $\text{sigmoid}$ 같은 활성함수를 통과하는 Non_linear 구간이 있습니다.

### Linear
Affine transform은 선형성을 가지는 대표적인 변환입니다. 선형성은 각 입력에 대해서, 입력값을 증가하면 다른 입력값과는 상관없이 결과값이 커지거나 작아지는 것을 의미합니다. input unit들의 영향력이라고 생각하면 됩니다.


### Non-Linear
Nonlinear 구간이 등장하는 가장 간단한 이유는 input units간의 interaction을 반영하기 위함입니다. 단순한 linear relation 만 고려한다면, 단순한 linear transform의 합으로 된다. 이러한 linear classifier 는 한계점이 존재하게 됩니다. 이러한 문제들은 Linear한 모델들을 생각하면 편합니다. 예를 들어 Logistic regression과 linear regression을 본다면, closed form solution이나 convex optimization을 가진다는 장점을 가지게 됩니다. 하지만 Linear model는 Linear function을 기반으로 한다는 한계점을 가지게 됩니다. input unit간의 interaction을 반영하지 못하는 결과를 낳게 됩니다.

만약 현실세계의 데이터를 받아본다면, 선형 classifier가 결정되긴 매우 힘들게 됩니다. 때문에 input들의 interaction을 고려할 수 있는 활성화 함수를 도입하게 됩니다. 활성화 함수 $\phi$를 사용하면 XOR 문제 등 비선형 문제를 해결 가능하게 됩니다. 흔히 사용하는 활성화 함수들은 nonlinear transformation의 기능을 지원하며, 이는 저희가 알고있는 kernel trick과 유사한 역할을 하게 됩니다. 

```{figure} ../images/activation.png
---
name: activation function
---
activation function, 출처: [Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning/home/welcome)
```

## output to input
일반적으로 뉴럴넷은 레이어가 진행되면서, Linear, Non-Linear를 구간을 거친 ouput 데이터가 ($a$) 다시 새로운 input으로 들어가게 됩니다.

```{figure} ../images/nn.png
---
name: Decision Boundary
---
Decision Boundary, 출처: [Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning/home/welcome)
```

위 그림을 보시면 linear한 classifier들이 Affine Transform 을 지나서 Activation을 통과하면 non-linear한 결정 경계가 만들어지는 것을 보실수 있습니다. 강조된 파란색 동그라미는 $\hat{y}$를 표현합니다.

## Ouput Layer
최종적으로 Output layer에서는 무엇을 산출하게 될까요? 그것은 우리가 풀려고 하는 Task에 따라서 달라지게 됩니다. 예를들어 맞냐 틀리냐와 같은 이진 분류 문제의 경우에는 0,1 로 결과물을 내야합니다. 이럴때는 output에 sigmoid를 넣고 theshold을 설정하게 됩니다. 만약에 다양한 선택지중 정답을 찾아야하는 multi-classification task의 경우에는 output에 softmax를 통해서 정답을 찾게 됩니다. 만약 수치를 찾게되는 Regression 문제의 경우에는 그냥 산출 output을 받으면 됩니다. 여기서 본다면 우리의 output layer 설정의 단서를 얻을 수 있습니다. Output layer에는 classification 하려는 class 갯수만큼 설정하게 됩니다.

$$
\text{Sigmoid} = \frac{1}{1+e^{-\theta x}} \\
\text{Softmax} = \frac{e^{\theta_1 x}}{\sum _{j}^{}{e^{\theta_j x_j}}} \\
\text{Linear} = W^{[L]}h^{[L-1]}+b^{[L]}
$$

## Loss function

Forward Propagation을 통과해서 나오는 $\hat{y}$은 모델의 산출물이긴 하지만 실제 데이터 셋에서 주어진 $x$에 대한 pair인 $y$와 다를 수 있습니다. 때문에 오차를 계산해주어야 합니다. 우리는 이 오차를 계산하는 함수를 Loss function이라고 합니다. 

## Code Implementation

코드 구현은 andrew ug 교수님의 coursera강의 중 Building your Deep Neural Network - Step by Step 과제로 표현합니다.

모형의 아키텍쳐는 2layer MLP로 구성되어있으며, hidden unit의 activation function을 relu 그리고 output layer의 activation function을 sigmoid로 설정되어 있습니다. 구조를 다음 그림으로 표현하겠습니다.


```{figure} ../images/arci.png
---
name: Two Layer MLP
---
Two Layer MLP, 출처: [Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning/home/welcome)
```

### 1. Architecture Setting

뉴럴넷의 가장 간단한 아키텍츠를 세팅해보려고 합니다. 저희에게 필요한 것은 레이어의 갯수와 뉴런의 갯수 그리고, 각 layer들의 연결 패턴을 고려해야합니다. 이번 케이스에서는 Input, hidden, output layer를 각각 하나씩 가지고 있는 모듈을 만들어 보려고합니다. connectivity pattern은 fully connected 로 하겠습니다. 이후 CNN과 RNN post를 통해 다른 connectivity pattern을 소개하겠습니다. 레이어와 뉴런의 갯수는 list의 길이와 element를 통해 표현이 가능합니다.

- Layer number
- Neuron number
- connectivity pattern


```python
layer_dims = [5,4,3,1]
```


```python
import numpy as np

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    cache = Z 
    return A, cache
```

### 2.Initialize_parameters

학습할 파라미터들의 초기값을 잡아주어야합니다. 뉴럴넷에서 학습이 이루어져야하는 파라미터는 일반적으로 weight와 bias입니다. weight와 bias의 초기값 설정은 매우 중요합니다. 일반적으로 가장 크게 문제가 되는 경우는 Gradient vanishing이나 local minmum에 빠지는 문제점이 발생하게 됩니다. 일반적으로는 활성화함수로 어떤것을 사용하느냐에 따라서 다른 초기화 방법론이 사용되기도 합니다.(ex. sigmoid, relu 등등) 

- Sigmoid, tanh : Xavier Initalization
- ReLU : He Initalization

우리의 목표는 각 input에 따라서 진행되는 weight가 뉴런마다 적당한 variance를 가지면서 좋은 classifier를 형성하는데 있습니다. 다양한 초기화 방법이 있으나 이후 포스트에서 다루기로 하고 이번에는 Gaussian 분포를 따르면서, 0.01값을 곱해서 매우 작은 값을 가지는 weigth를 이용하여 초기화를 하려고 합니다. 가중치의 초기값은 매우 작은값으로 설정하는 것이 좋습니다. 왜냐하면, 가중치가 너무 큰 값을 가지는 경우 활성값을 계산하게 된다면, 활성화 함수가 거의 1 또는 0으로 확실하게 적용되기 떄문입니다. 이는 학습이 느려지는 문제를 가지게 됩니다. 해당 난수의 생성은 numpy패키지의 np.random.randn를 사용하여 해보도록하겠습니다.

__Arguments__<br>
layer_dims : Input으로 각 레이어들의 dimension을 받게 됩니다. Dimension이라는 것은 각 레이어별로 존재하는 node의 갯수라고 생각하시면 좋습니다.

__Returns__<br>
parameters : python dictionary를 return합니다. 
key는 각 weight와 bias의 string이 들어가게 되며 ("W1", "b1", ..., "WL", "bL"), value에는 해당 값들이 들어가게 됩니다.
- Wl : weight matrix입니다. shape은 두개의 레이어의 디멘션으로 구성이 됩니다. (layer_dims[l], layer_dims[l-1])
- bl : 해당 레이어의 bias들입니다. (layer_dims[l], 1)


```python
def initialize_parameters_deep(layer_dims):
    # dictionary 객체 생성
    parameters = {}
    # 총 layer들의 길이를 계산
    L = len(layer_dims)
    # 레이어들을 돌면서, 레이어들 간의 weight와 bias의 초기값의 난수 생성
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        # assert를 통해, dimension을 맞추줍니다. 틀릴시 error 발생
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        
    return parameters
```


```python
parameters = initialize_parameters_deep(layer_dims)
parameters
```




    {'W1': array([[ 0.34522571, -0.00435798, -0.55015057, -0.18029009, -0.24863248],
            [ 0.48145223,  0.48976974,  0.4544385 ,  0.70003399,  0.45756044],
            [-0.55900747, -0.09964053,  0.28876994,  0.76200738, -0.54389478],
            [ 0.07710557, -0.27164855, -0.57329413,  0.56887702,  0.45411553]]),
     'b1': array([[0.],
            [0.],
            [0.],
            [0.]]),
     'W2': array([[-0.02059886,  0.19182803,  0.11765535,  0.0742146 ],
            [-0.75716588,  0.93100703,  0.56786299, -0.65197777],
            [-0.55100467, -0.26469174, -0.21975884, -0.01100711]]),
     'b2': array([[0.],
            [0.],
            [0.]]),
     'W3': array([[ 0.93923832,  1.23714427, -0.18204054]]),
     'b3': array([[0.]])}



### 3.Forward Propagation

일반적으로 forward 패스는 linear 부분과 non-linear부분으로 나누어져서 진행되게 되어있다. 이번에 구현한 구조는 input layer에서 hidden layer로 가면서는 relu activation function을 그리고, hidden을 지나 output으로 이동하면서 sigmoid activation function을 사용하는 모형이다. 이 모형은 크게 linear 한 연산을 하는 Linear_forwar과 non-linear한 연산을 하는 Linear-Activation Forward으로 나눌수 있다.


#### Linear_forward

__Arguments__

A : input data를 받거나 혹은, 이전 단계에서 activation 함수를 통과한 함수가 됩니다. 벡터의 사이즈는 input number이거나, 이전 레이어의 node만큼 들어오게 됩니다.

W : weights matrix 입니다. numpy array of shape으로 구성되어 있습니다. (size of current layer, size of previous layer)

b : bias vector입니다. numpy array of shape으로 구성되어 있습니다. (size of the current layer, 1)

__Returns__

Z : affine transform을 지나서 나온 ouput입니다. 이것이 Activation funtion의 input으로 들어가게 됩니다.

cache : python dictionary입니다. "A", "W" 그리고 "b"의 값을 저장합니다 이는 이후의 backward 상황에서 효율적인 계산을 도와주게 됩니다.

$$Z^{[l]} = W^{[l]}A^{[l-1]} +b^{[l]}$$

#### Linear-Activation Forward

Activation function의 경우에는 위에서 작성한 Linear_forward를 받아서 가지고 옵니다. 또한 이때 Linear 계산값과, Activation을 통과한 값 모두 backward과정에서 함수 값을 다시 사용하는 경우가 있기 떄문에 cache에 따로 저장을 해둡니다.


__Arguments__<br>
A_prev : input data를 받거나 혹은, 이전 단계에서 activation 함수를 통과한 함수가 됩니다. 벡터의 사이즈는 input number이거나, 이전 레이어의 node만큼 들어오게 됩니다. A_prev, W, b 모두 linear_forward와 같은 인자입니다.<br>
W : weights matrix 입니다. numpy array of shape으로 구성되어 있습니다. (size of current layer, size of previous layer)<br>
b : bias vector입니다. numpy array of shape으로 구성되어 있습니다. (size of the current layer, 1)


__Returns__<br>
A : Activation function을 통과한 output입니다. 이것은 다음 layer의 input으로 들어가게 됩니다.<br>
cache : python dictionary형태입니다. backward pass의 효율적인 계산을 위해서, affine transform을 거친 값과, activation function을 거친 값 모두 저장해둡니다. key는 "linear_cache" ,"activation_cache"에 각각의 value를 저장해 둡니다.
- Sigmoid
$$\sigma(Z) = \sigma(W A + b) = \frac{1}{ 1 + e^{-(W A + b)}}$$
- ReLU 
$$A = RELU(Z) = max(0, Z)$$.
<br>

```python
def linear_forward(A, W, b):
    # W에 A를 내적하게 됩니다. 그후에는 b를 더해줍니다.
    Z = W.dot(A) + b
    # Z의 shape이 input과 weight의 shape과 동일한지를 체크합니다.
    assert(Z.shape == (W.shape[0], A.shape[1]))
    # 계산단계에서 사용한 값을 cache에 저장해둡니다.
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    # Activation function의 종류에 따라서 값을 나누어 줍니다.
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    # Shape이 input과 weight와 동일한지 체크해줍니다.
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    # linear 연산과 activation 연산을 cache에 저장해둡니다.
    cache = (linear_cache, activation_cache)
    return A, cache
```

#### L-Layer Model 

이제 위에서 정의했던 linear 함수들을 조합하여 새로운 L-layer model을 만들어 봅시다.

__Arguments__<br>
X : input으로 들어오게 될 data 의 matrix입니다. (input size, number of examples)<br>
parameters : 초기에 선언해 둔 parameter들입니다. 위에서 선언한 initialize_parameters_deep의 결과값입니다.

__Returns__<br>
AL : 마지막 layer의 activation 값입니다.<br>
caches : 모든 linear 부분의 결과 값들과 (there are L-1 of them, indexed from 0 to L-2), 모든 non-linear들의 계산의 결과값들이 저장된 dictionary들이 담긴 list입니다. (there is one, indexed L-1)
<br>

```python
def L_model_forward(X, parameters):
    # cache 들의 list입니다.
    caches = []
    A = X
    # weight와 bias가 저장되어 있기 때문에 //2 를 해주어야 layer의 사이즈가 됩니다.
    L = len(parameters) // 2
    
    # hidden layersms relu를 통과
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)
    
    # output layer는 sigmoid를 통과하게 한다
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
    caches.append(cache)
    assert(AL.shape == (1,X.shape[1]))
    return AL, caches
```


```python
X = np.random.randn(5,4)
Y = np.array([[0, 1, 1, 0]])
AL, caches = L_model_forward(X, parameters)
```


```python
AL
```




    array([[0.50806541, 0.50446785, 0.99754058, 0.5       ]])



### 4. Cost Function

우리는 신경망을 통과한 $\hat{y}$값을 찾을 수 있었습니다. 하지만 우리의 실제 y 레이블과는 다른 값일 가능성이 매우 크기 떄문에 이를 반영하여 학습을 시켜야합니다. Cost function은 여러가지 종류가 있습니다만, 이번의 경우에는 cross-entropy 함수를 사용하려고합니다. 이후에 Cost function에 대해서도 정리해보도록 하겠습니다.

$$
-\frac{1}{m} \sum\limits_{i = 1}^{m} (y^{(i)}\log\left(a^{[L] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right))
$$


__Arguments__<br>
AL : 뉴럴넷을 통과해서 나오게된 $\hat{y}$ 입니다. shape (1, number of examples)<br>
Y -- 실제 "label" vector 입니다. (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

__Returns__<br>
cost : cross-entropy cost
<br>

```python
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (-1.0/m)*np.sum(np.multiply(Y,np.log(AL)) + np.multiply(1-Y, np.log(1-AL)))
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    return cost
```


```python
cost = compute_cost(AL, Y)
print("cost = " + str(cost))
```

    cost = 0.5223175799014049


### 5.Backward propagation

이제 저희는 Cost function에서 보면 변수는 $a^{[L] (i)}$입니다. 그리고 $a$를 구성하는 가장 parameter는 weight와 bais입니다. 따라서 Cost function 은 W와 b에 대해서 구성되었다고 말할 수 있습니다. 

Backward propagation의 가장 직관적인 이해는 제가 생각할때, Cost function중 해당 parameter가 기여한 부분만큼을 계산하여 updata에 반영하는 것입니다. 따라서 편미분의 개념이 사용되는 것이죠. 편미분은 다른 변수를 고정시키고 특정 변수에 대해서 미분이 진행되는 것입니다. 따라서 parameter 하나하나에 대해서 Cost function의 변화량에 기여하는 부분만큼을 고려하여 미분을 진행시키는 것입니다.

Andrew ug 교수님은 다음과 같은 그림을 통해서 backward propagation을 설명해주시고 있습니다.


```{figure} ../images/back.png
---
name: Backward propagation
---
Backward propagation, 출처: [Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning/home/welcome)
```

$$\frac{d \mathcal{L}(a^{[2]},y)}{{dz^{[1]}}} = \frac{d\mathcal{L}(a^{[2]},y)}{{da^{[2]}}}\frac{{da^{[2]}}}{{dz^{[2]}}}\frac{{dz^{[2]}}}{{da^{[1]}}}\frac{{da^{[1]}}}{{dz^{[1]}}} $$


Gradient를 계산하는 방식은 다음과 같습니다. Loss 값에 대한 $dA$, 즉 Non-linear 파트에 대한 Gradient를 계산하고, 그 다음은 $dZ$, linear 파트에 대한 Gradient를 계산하고, 그 이후에 $dW$와 $db$를 계산해주면서 넘어가게 됩니다.

간단한 설명을 위해 logistic regression 으로 돌아가서 생각해 봅시다. 저희가 $da$를 계산했던 방식은 loss function을 da에 대해서 미분해준 값이였습니다. 

$$da = \frac{d \mathcal{L}(a,y)}{{da}} = \frac{-y}{a}+ \frac{1-y}{1-a}$$

그 다음 계산이 이루어진 부분은 non-linear 파트와 linear 파트를 당담하는 부분입니다. 저희는 chain rule을 사용해서 한번에 계산이 이루어 졌었습니다. actvation function을 $g(z)$라고 둔다면 저희의 $dz$의 값은, $da$을 받아서 미분된 activation에 $z$값을 넣어서 계산된량을 곱해진 값이 됩니다.

$$
dz = da \times g'(z) \\
\frac{\partial L}{\partial z} = \frac{\partial L}{\partial a} \times \frac{da}{dz} \\
\frac{da}{dz} = \frac{d}{dz}g(z) = g'(z)
$$

$dW$를 설명하는 식은 ,Loss function에 대한 W의 미분으로 표현되나 이는 chain rule을 계산해 본다면, 이전 step 의 input에 $dz$ 를 넣어준 값이 됩니다.

$$
dW = \frac{\partial L}{\partial W} \\
dW = dz \times \frac{\partial z}{\partial W} = dz \times a^{l-1}
$$

동일한 방식으로 bias에 대해서 gradient를 계산한다면 
$$
db = \frac{\partial L}{\partial b}\\
db = dz \times \frac{\partial z}{\partial b} = dz
$$

위 notation을 본다면 우리는 이것이 recursive한 형태로 주어진다는 것을 알게됩니다. 여기서 dynamic programming의 필요성도 나오게 되는 것이 되는것입니다. 또한 위의 계산 방식을 본다면, 두가지 재밌는 성질을 볼 수 있습니다. 

첫번쨰는 계산의 그래프의 앞 순서에서 표현됬던, 값들이 전파되는 것을 볼수 있습니다. 예를들어 $dz$를 계산하는데 $da$가 사용되며, $dw$와 $db$를 계산하는데 $dz$가 사용되는것이 그 예시입니다. 두번째는 forward에서 linear 파트에서 계산되었던 값들이 activation function의 미분함수의 input으로 들어가는 점입니다. 또한 forward에서 non-linear에서 계산되었던 값들이 $dw$를 계산하는 과정에서 사용된다는 점입니다.

__Backpropagation Process__<br>
Backpropagation은 다음과 같은 process를 가지게 됩니다.
- LINEAR backward
- LINEAR -> ACTIVATION backward
- Layer -> Layer backward

### Linear backward
Linear 한 영역에서 backward 과정은 다음과 같은 인자를 받게 됩니다.
__Arguments__<br>
dZ : Z의 변화량입니다. linear 부분에서 ouput이 cost function 에 대한 gradient를 나타냅니다.<br>
cache : forward과정에서 필요한 값을 받아옵니다. tuple 형태의 (A_prev, W, b) 값들을 받아옵니다.

__Returns__<br>
dA_prev : Linear 구간의 input으로 들어왔었던, 지난 레이어의 activation 을 통과한 A가 cost function에 대한 변화량입니다.<br>
dW : Linear 구간의 weight의 cost function에 대한 변화량 입니다.<br>
db : Linear 구간의 bias의 cost function 에 대한 변화량 입니다.

#### Linear-Activation backward
Activation function $g(.)$ 에 대해서 Linear-activate backward는 다음과 같이 계산됩니다.

$$
dZ^{[l]} = dA^{[l]} * g'(Z^{[l]})
$$  

__Arguments__<br>
dA : 현재 layer의 gradient값이 인자로 들어옵니다.<br>
cache : forward pass에서 계산했던 linear(Z) 부분과 activation(A) 부분의 계산값들을 받습니다.

__Returns__<br>
dA_prev : Linear 구간의 input으로 들어왔었던, 지난 레이어의 activation 을 통과한 A가 cost function에 대한 변화량입니다.<br>
dW : Linear 구간의 weight의 cost function에 대한 변화량 입니다.<br>
db : Linear 구간의 bias의 cost function 에 대한 변화량 입니다.<br>

$$ 
dW^{[l]} = \frac{\partial \mathcal{L} }{\partial W^{[l]}} = \frac{1}{m} dZ^{[l]} A^{[l-1] T}
$$
$$
db^{[l]} = \frac{\partial \mathcal{L} }{\partial b^{[l]}} = \frac{1}{m} \sum_{i = 1}^{m} dZ^{[l](i)}
$$
$$dA^{[l-1]} = \frac{\partial \mathcal{L} }{\partial A^{[l-1]}} = W^{[l] T} dZ^{[l]}
$$
```python
def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    return dZ
```
```python
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = np.dot(dZ,cache[0].T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(cache[1].T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db =  linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db
```

#### L-Model Backward 
이제 위에서 정의했던 backward 함수들을 을 조합하여 새로운 L-layer model을 만들어 봅시다.

__Initializing backpropagation__<br>
가장먼저 Output layer에서 가장, 끝부분에 있었던 산출물의 loss를 구해줍니다.
$$
dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
$$

우리의 Gradient는 학습의 핵심이 되는 weight와 bias를 업데이트하는 중요한 역할을 하게 됩니다. 때문에 해당 Weight Gradient값을 dictionary 형태로 저장해 줍시다/

$$
grads["dW" + str(l)] = dW^{[l]}
$$

__Arguments__<br>
AL : 확률 벡터입니다. Forward propagtaion의 최종 아웃풋이기도 합니다.<br>
Y : 실제 y, label 값입니다.

__Returns__<br>
grads : Layer 별로 Gradient 값들이 저장되어있는 dictionary입니다
- grads["dA" + str(l)] = ... 
- grads["dW" + str(l)] = ...
- grads["db" + str(l)] = ... 


```python
def L_model_backward(AL, Y, caches):
    grads = {} # 빈 dictionary 호출
    L = len(caches) # 레이어의 갯수를 caches로 부터 받아옵니다.
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # Shape을 AL과 동일하게 해줍니다.
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y,AL)- np.divide(1-Y, 1-AL))
    # caches index를 잡아둡니다.
    current_cache = caches[L-1] 
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation="sigmoid")
    
    for l in reversed(range(L-1)):
        # indexing입니다.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+1)], current_cache, activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads
```


```python
grads = L_model_backward(AL, Y, caches)
grads
```




    {'dA2': array([[ 4.77194502e-01, -4.65422790e-01, -2.30997864e-03,
              4.69619161e-01],
            [ 6.28550208e-01, -6.13044765e-01, -3.04265357e-03,
              6.18572134e-01],
            [-9.24884992e-02,  9.02069390e-02,  4.47713578e-04,
             -9.10202679e-02]]),
     'dW3': array([[-0.00229851,  0.00028332,  0.        ]]),
     'db3': array([[0.12751846]]),
     'dA1': array([[-0.48574643,  0.00958718,  0.00235138,  0.        ],
            [ 0.67672394, -0.08928114, -0.00327585,  0.        ],
            [ 0.41307489, -0.05475948, -0.00199959,  0.        ],
            [-0.37438596, -0.03454117,  0.00181231,  0.        ]]),
     'dW2': array([[-8.23189416e-02, -1.42110237e-03,  2.79039103e-03,
             -5.26880192e-02],
            [ 0.00000000e+00, -1.87184509e-03,  3.67544230e-03,
             -9.30159738e-06],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
              0.00000000e+00]]),
     'db2': array([[0.00236543],
            [0.15637689],
            [0.        ]]),
     'dA0': array([[-0.23091195,  0.00064642, -0.00031964,  0.        ],
            [-0.041159  ,  0.00934128, -0.00189748,  0.        ],
            [ 0.11928361,  0.01452786, -0.00310508,  0.        ],
            [ 0.31476611, -0.02137815, -0.00278593,  0.        ],
            [-0.22466927, -0.01806936,  0.00041167,  0.        ]]),
     'dW1': array([[-0.00156397, -0.00058392, -0.00353235, -0.00187307,  0.00019292],
            [-0.00053448,  0.00106037, -0.00222424, -0.00239189,  0.00089139],
            [-0.06663642, -0.07072757,  0.15135499, -0.05993305,  0.07339559],
            [ 0.00593044,  0.00151714,  0.01395705,  0.00807165, -0.00118819]]),
     'db1': array([[ 0.00239679],
            [-0.00081896],
            [ 0.10276882],
            [-0.00818221]])}



### 6. Update parameter

파라미터를 업데이트 하는 규칙은 생각보다 간편합니다. Learning rate인 $\alpha$ 에 Gradient를 곱해서 현재의 parameter에 빼주면 새로운 parameter가 됩니다.

$$
W^{[l]} = W^{[l]} - \alpha \text{ } dW^{[l]}
$$
$$
b^{[l]} = b^{[l]} - \alpha \text{ } db^{[l]}
$$

__Arguments__<br>
parameters : 파라미터들이 담겨져 있는 parameter dictionary입니다.
grads : Gradient들이 담겨있는 입니다

__Returns__<br>
parameters : 업데이트되어있는 파라미터들이 담긴 dictionary입니다
- parameters["W" + str(l)] = ... 
- parameters["b" + str(l)] = ...


```python
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 # 레이어의 갯수입니다.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW"+str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db"+str(l+1)]
    return parameters
```


```python
parameters = update_parameters(parameters, grads, 0.05)
parameters
```




    {'W1': array([[ 0.34538211, -0.00429959, -0.54979734, -0.18010278, -0.24865177],
            [ 0.48150568,  0.48966371,  0.45466093,  0.70027318,  0.4574713 ],
            [-0.55234383, -0.09256777,  0.27363445,  0.76800069, -0.55123434],
            [ 0.07651252, -0.27180027, -0.57468984,  0.56806986,  0.45423435]]),
     'b1': array([[-2.39679482e-04],
            [ 8.18962628e-05],
            [-1.02768824e-02],
            [ 8.18221459e-04]]),
     'W2': array([[-0.01236697,  0.19197014,  0.11737631,  0.0794834 ],
            [-0.75716588,  0.93119421,  0.56749545, -0.65197684],
            [-0.55100467, -0.26469174, -0.21975884, -0.01100711]]),
     'b2': array([[-0.00023654],
            [-0.01563769],
            [ 0.        ]]),
     'W3': array([[ 0.93946817,  1.23711594, -0.18204054]]),
     'b3': array([[-0.01275185]])}
