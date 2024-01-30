# Training Parallelism Overview (Data, Model, Tensor)

- 이 글은 [Everything about Distributed Training and Efficient Finetuning-Sumanthrh](https://sumanthrh.com/post/distributed-and-efficient-finetuning/), [How to Train Really Large Models on Many GPUs?-lilianweng](https://lilianweng.github.io/posts/2021-09-25-train-large/) [TUNiB-large-scale-lm-tutorials](https://github.com/tunib-ai/large-scale-lm-tutorials)을 참고했습니다.

요즘 Large-Scale Training은 모델의 성능과 새로운 [Emerging ability](https://arxiv.org/abs/2206.07682)를 보여주기 떄문에 많은 사람들의 관심을 받고 있습니다. 이러한 Large-Scale 학습의 목적은 큰 모델 크기 (Billion Scale)과 대규모 데이터셋 (Million Scale) 입니다. 우리의 목표는 가능한 많은 샘플을 초당 처리할수 있도록 하는 것입니다. 이러한 문제로 인해 우리는 큰 GPU vRAM이 필요합니다. 모델 가중치와 데이터를 올려야합니다. 

이렇듯 큰 모델과 큰 데이터를 처리하려면 여러대의 GPU를 사용하여 병렬 처리 (Parallelism)가 중요합니다. 병렬처리는 여러개를 동시에 처리하는 기술을 뜻하며, 이떄 중요한 것은 데이터 처리량 (throughput), 메모리의 효율성 등입니다. 병렬 처리 (Parallelism)는 모델의 측면 (MP,PP), 데이터의 측면 (DP), 그리고 텐서의 측면 (TP)에서 진행할 수 있습니다. 

## Data Parallelism(DP, DDP)

Data Parallelism은 엄격하게 DP와 DDP로 나누어집니다. DP는 0번째 GPU에서 모델의 업데이트가 이루어지고, 모델은 각 GPU로 복제됩니다. forward 과정에서 각 GPU의 output들은 loss연산을 위해서 GPU 0번으로 보내지고, 연산된 loss는 다시 각 GPU로 분산됩니다. 또한 각 GPU의 gradient또한 GPU 0번으로 보내져서 평균화 됩니다. 이 과정에서 GPU 0은 큰 메모리 소모가 발생하게 됩니다. 

1. Data Scatter: Bactch를 Scatter하여 각 GPU로 전송
2. Model Broadcast: GPU 0번째 모델을 다른 GPU로 Broadcase
3. **Forward**: 각 Broadcase된 모델들은 Scatter된 데이터를 처리하여 Output 연산
4. Output Gather: Output을 다시 GPU 0번에 모음 & Loss연산
5. Loss Scatter: 계산된 Loss를 각 GPU에 Scatter
6. **Backward**: 각 GPU는 Gradients 연산 (`loss.backward()`)
7. Gradient Reduce: 계산된 Gradient를 GPU 0번에서 Reduce하여 더함
8. Update: 더해진 Graident들을 사용해서 GPU 0번째 모델을 업데이트 `optimizer.step()`

**backward()가 step()보다 무거운 연산임을 알아두면 좋습니다.**

이와 달리 DDP는 각 GPU에 모델을 복제하고 각각의 로컬 그라디언트를 모든 프로세스에서 평균화하는 방식을 사용합니다. 이는 그라디언트를 효율적으로 통합하여 학습을 진행합니다. 또 다른 차이점은 DDP는 torch.distributed를 사용하여 데이터를 복사하고, DP는 Python 스레드를 통해 프로세스 내에서 데이터를 복사합니다(이로 인해 GIL과 관련된 제한이 발생합니다). 결과적으로 Distributed Data Parallel (DDP)이 일반적으로 Data Parallel (DP)보다 빠르며, 느린 GPU 간 연결이 없는 경우에 해당합니다.

물론 DDP가 효율적이며, 일반적으로 Data Parallelism을 뜻하게 됩니다. 

각 GPU 워커는 전체 미니 배치의 일부를 받아들이고, 해당 미니 배치 데이터에서 Gradients를 계산합니다. 그런 다음 Gradients는 모든 워커를 통해 평균화되고 모델 가중치가 업데이트됩니다. PyTorch DDP와 같은 가장 기본적인 형태에서 각 GPU는 작업 중인 데이터 부분에 대한 weight, optimizer, gradients의 status 저장합니다.

## Model Parallelism / Pipeline Parallelism (MP / PP)

모델이 너무 커서 하나의 디바이스에 올라갈수 없다면, 여러 GPU에 나눠서 올려야합니다. 모델 병렬화에서는 모델이 수직 혹은 수평으로 나뉘어 모델의 다른 레이어가 다른 GPU 워커에 배치됩니다. 단일 모델에 12개의 레이어가 있는 경우 3개의 GPU에 배치된 경우를 생각해보세요. 

```bash
---------------  ---------------  -----------------
1 | 2 | 3 | 4 |  5 | 6 | 7 | 8 |  9 | 10 | 11 | 12 |
---------------  ---------------  -----------------
```

단순 모델 병렬화에서는 모든 GPU가 동일한 데이터 배치를 처리하는 동안 이전 GPU가 계산을 마칠 때까지 기다린 다음 데이터를 처리합니다. 이를 개선한 것이 바로 Pipeline Parallelism (PP) 입니다. 여기서는 Micro Batch를 구현하여, 마이크로 배치 데이터에 대해 계산을 겹쳐서 병렬처리를 구현합니다.

## Tensor Parallelism (TP)

텐서 병렬화에서 각 GPU는 모델을 수평으로 나누어 텐서의 일부만 처리합니다. 각 워커는 동일한 데이터 배치를 처리하며 자신이 가진 가중치 부분에 대한 활성화를 계산하고 서로 필요한 부분을 교환하며 각 워커는 가중치 일부에 대한 기울기를 계산합니다.

```
import torch
import torch.nn.functional as F

### standard mlp
x = torch.randn(1,16)
w1 = torch.randn(16,64)
w2 = torch.randn(64,16)
act_fn = F.relu
y = act_fn(x @ w1) @ w2

### column parallel
w11, w12 = torch.split(w1, split_size_or_sections=[32,32], dim=1)
# row parallel
w21, w22 = torch.split(w2, split_size_or_sections=[32,32], dim=0)
y_test = act_fn(x @ w11) @ w21 + act_fn(x @ w12) @ w22
torch.allclose(y, y_test)
```