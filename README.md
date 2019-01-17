# Overcoming-Catastrophic-forgetting-in-Neural-Networks
Elastic weight consolidation technique for incremental learning.
!! Testing in progress !!
## About
Use this API if you dont want your neural network to forget previously learnt tasks while doing transfer learning or domain adaption!
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
![Paper](https://arxiv.org/abs/1612.00796)

