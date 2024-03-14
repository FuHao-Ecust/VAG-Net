# Abdominal multi-organ segmentation in Multi-sequence MRIs  based on visual attention guided network and knowledge distillation


# Dependencies
You want to install the usual, pytorch, tqdm, pandas. Also, the architectures I use belong to the pytorch-segmentation-models library, which can be installed via pip install segmentation-models-pytorch.

# Data availability
We evaluate our proposed method on the CHAOS 2019 Challenge dataset. You can head to [grand-challenge](https://chaos.grand-challenge.org/Data/), where you will be able to download the training data, Validation and test data.

# Running experiments - Training

```
python train_kdloss.py
```

# Running experiments - predicting

```
python predict.py
```
