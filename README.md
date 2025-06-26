# SGFormer-RGCN: Accurate Performance Prediction for High-Level Synthesis Design Space Exploration

## Content
- [About the project](#about-the-project)
- [Required environment](#required-environment)
- [Baseline](#baseline)
## About the project
we propose a high-precision performance prediction model, SGFormer-RGCN, that integrates both global and local modeling capabilities. The model organically combines the global contextual representation strength of SGFormer with the local structural awareness of the RGCN. By introducing a learnable adaptive fusion mechanism, the model dynamically allocates weights between global and local features, enabling more flexible and efficient information integration. This mechanism not only captures long-range dependencies but also identifies fine-grained structural patterns. 
![SGFormer-RGCN Architecture](framework/framework.pdf)


### Contribution




## Required environment
### Python Environmant
- python 3.9
- torch 1.13.1
- torch-geometric 2.3.1
- torch-scatter 2.1.1
- torch-sparse 0.6.17
- optuna 3.2.0
- pyDOE 0.3.8
### Vitis Environment
- Vitis HLS 2022.1
- Vivado 2022.1

## Baseline
- IRONMAN-PRO
- GCN+GF
- GAT+GF
- SAGE+GF
- RGCN+GF
