**Note:** The code is currently in an experimental version. 

#### Model
The "STab" folder contains all model-related files.

#### Examples
We provide Jupyter notebooks for training the model with each of the studied datasets.

#### Data
To download the same versions and splits for YE, HI, HE, AD, and JA, please follow the instructions in this repository:
[Download Link](https://www.dropbox.com/s/o53umyg6mn3zhxy/data.tar.gz?dl=1) -O revisiting_models_data.tar.gz

To download the same versions and splits for DI, HO, and OT, please follow the instructions in this repository:
[Download Link](https://huggingface.co/datasets/puhsu/tabular-benchmarks/resolve/main/data.tar) -O tabular-dl-tabr.tar.gz

To use the example notebooks, copy the dataset folder directly into the "Data" directory. 

#### Dependencies
We use torch for model implementation and the keras4torch library for easier training and evaluation. Additionally, our code makes use of elements from the following repositories:
- [FtTransformer PyTorch](https://github.com/lucidrains/tab-transformer-pytorch)
- [Stochastic Transformer Networks with Linear Competing Units](https://github.com/avoskou/Stochastic-Transformer-Networks-with-Linear-Competing-Units-Application-to-end-to-end-SL-Translation)
- [Keras 4 Torch](https://github.com/blueloveTH/keras4torch)
