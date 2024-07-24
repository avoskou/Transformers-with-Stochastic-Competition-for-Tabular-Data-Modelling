# Transformers with Stochastic Competition for Tabular Data Modelling


Despite the prevalence and significance of tabular data across numerous industries and fields, it has been relatively underexplored in the realm of deep learning. Even today, neural networks are often overshadowed by techniques such as gradient boosted decision trees (GBDT). However, recent models are beginning to close this gap, outperforming GBDT in various setups and garnering increased attention in the field. Inspired by this development, we introduce a novel stochastic deep learning model specifically designed for tabular data. The foundation of this model is a Transformer-based architecture, carefully adapted to cater to the unique properties of tabular data through strategic architectural modifications and leveraging two forms of stochastic competition. First, we employ stochastic "Local Winner Takes All" units to promote generalization capacity through stochasticity and sparsity. Second, we introduce a novel embedding layer that selects among alternative linear embedding layers through a mechanism of stochastic competition. The effectiveness of the model is validated on a variety of widely-used, publicly available datasets. We demonstrate that, through the incorporation of these elements, our model yields high performance and marks a significant advancement in the application of deep learning to tabular data.


## Reference
Please cite:

@misc{voskou2024transformersstochasticcompetitiontabular,
      title={Transformers with Stochastic Competition for Tabular Data Modelling}, 
      author={Andreas Voskou and Charalambos Christoforou and Sotirios Chatzis},
      year={2024},
      eprint={2407.13238},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.13238}, 
}


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
