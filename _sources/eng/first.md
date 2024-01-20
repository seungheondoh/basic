# Training Parallelism Overview (Data, Model, Tensor)

- 이 글은 [Everything about Distributed Training and Efficient Finetuning-Sumanthrh](https://sumanthrh.com/post/distributed-and-efficient-finetuning/), [How to Train Really Large Models on Many GPUs?-lilianweng](https://lilianweng.github.io/posts/2021-09-25-train-large/) [TUNiB-large-scale-lm-tutorials](https://github.com/tunib-ai/large-scale-lm-tutorials)을 참고했습니다.

요즘 Large-Scale Training은 모델의 성능과 새로운 [Emerging ability](https://arxiv.org/abs/2206.07682)를 보여주기 떄문에 많은 사람들의 관심을 받고 있습니다. 이러한 Large-Scale 학습의 목적은 큰 모델 크기 (Billion Scale)과 대규모 데이터셋 (Million Scale) 입니다. 우리의 목표는 가능한 많은 샘플을 초당 처리할수 있도록 하는 것입니다. 이러한 문제로 인해 우리는 큰 GPU vRAM이 필요합니다. 모델 가중치와 데이터를 올려야합니다. 

이렇듯 큰 모델과 큰 데이터를 처리하려면 여러대의 GPU를 사용하여 분산처리가 중요합니다. 이때 분산처리는 모델의 측면에서도, 데이터의 측면에서도 진행할 수 있습니다.

1. Data Parallelism(DP, DDP): 각 GPU 워커는 전체 미니 배치의 일부를 받아들이고 해당 데이터 부분에서 Gradients를 계산합니다. 그런 다음 Gradients는 모든 워커를 통해 평균화되고 모델 가중치가 업데이트됩니다. PyTorch DDP와 같은 가장 기본적인 형태에서 각 GPU는 작업 중인 데이터 부분에 대한 모델 가중치, 옵티마이저 상태 및 Gradients의 복사본을 저장합니다.

2. (Vertical) Model Parallelism (MP): 모델 병렬화에서는 모델이 수직으로 나뉘어 모델의 다른 레이어가 다른 GPU 워커에 배치됩니다. 단일 모델에 12개의 레이어가 있는 경우 3개의 GPU에 배치된 경우를 생각해보세요. 

```bash
---------------  ---------------  -----------------
1 | 2 | 3 | 4 |  5 | 6 | 7 | 8 |  9 | 10 | 11 | 12 |
---------------  ---------------  -----------------
```

단순 모델 병렬화에서는 모든 GPU가 동일한 데이터 배치를 처리하는 동안 이전 GPU가 계산을 마칠 때까지 기다린 다음 데이터를 처리합니다. 이를 개선한 것이 바로 Pipeline Parallelism (PP) 입니다. 여기서는 Micro Batch를 구현하여, 마이크로 배치 데이터에 대해 계산을 겹쳐서 병렬처리를 구현합니다.

3. Tensor Parallelism (TP): 텐서 병렬화에서 각 GPU는 모델을 수평으로 나누어 텐서의 일부만 처리합니다. 각 워커는 동일한 데이터 배치를 처리하며 자신이 가진 가중치 부분에 대한 활성화를 계산하고 서로 필요한 부분을 교환하며 각 워커는 가중치 일부에 대한 기울기를 계산합니다.





