# Overcoming-Catastrophic-forgetting-in-Neural-Networks
Elastic weight consolidation technique for incremental learning.
## About
Use this API if you dont want your neural network to forget previously learnt tasks while doing transfer learning or domain adaption!
## Results
The experiment is done as follow:</br>
1. Train a 2 layer feed forward neural network on MNIST for 4 epochs
2. Train the same network later on Fashion-MNIST for 4 epochs
This is done once with EWC and then without EWC and results are calculated on test data for both data on same model. Constant learning rate of 1e-4 is used throughout with Adam Optimizer. Importance multiplier is kept at 10e5 and sampling is done with half data before moving to next dataset</br>

| EWC | MNIST | Fashion-MNIST |
| --- | ----- | ------------- |
| Yes | 70.27 |     81.88     |
| No  | 48.43 |     86.69     |
## Usage
```python
from elastic_weight_consolidation import ElasticWeightConsolidation
# Build a neural network of your choice and pytorch dataset for it
# Define a criterion class for new task and pass it as shown below
ewc = ElasticWeightConsolidation(model, crit, lr=0.01, weight=0.1)
# Training procedure
for input, target in dataloader:
  ewc.forward_backward_update(input, target)
ewc.register_ewc_params(dataset, batch_size, num_batches_to_run_for_sampling)
# Repeat this for each new task and it's corresponding dataset
```
## Reference
[Paper](https://arxiv.org/abs/1612.00796)

