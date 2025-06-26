# SGFormer-RGCN: Accurate Performance Prediction for High-Level Synthesis Design Space Exploration

## Content
- [About the project](#about-the-project)
- [Required environment](#required-environment)
- [Dataset](#dataset)
- [Baseline](#baseline)
## About the project
we propose a high-precision performance prediction model, SGFormer-RGCN, that integrates both global and local modeling capabilities. The model organically combines the global contextual representation strength of SGFormer with the local structural awareness of the RGCN. By introducing a learnable adaptive fusion mechanism, the model dynamically allocates weights between global and local features, enabling more flexible and efficient information integration. This mechanism not only captures long-range dependencies but also identifies fine-grained structural patterns. 
![SGFormer-RGCN Architecture](framework/framework.png)


### Contribution
- We adopt RGCN as the core predictive module, combined with a global pooling mechanism for feature integration. RGCN effectively leverages multi-relational edge types during message propagation, enhancing the model's ability to capture semantic dependencies among nodes. Compared to hierarchical pooling methods, global pooling does not require the introduction of additional node or edge selection mechanisms, thereby preserving the complete structural information of the graph.
- To overcome the limitation of conventional GNNs in modeling long-range dependencies, we propose a hybrid architecture SGFormer-RGCN. This model incorporates the SGFormer module, which employs a global attention mechanism with linear computational complexity to enable full-node interaction across the graph. By combining the local multi-relation aggregation capabilities of RGCN with the global dependency modeling of SGFormer, SGFormer-RGCN jointly captures both fine-grained local structures and holistic semantic patterns, significantly improving prediction accuracy and generalization capability.
- Experimental results show that SGFormer-RGCN achieves an average prediction error of 2.70\%–7.18\% across key performance metrics, including resource utilization, critical path timing, and power consumption, significantly outperforming existing methods. These results demonstrate the model's superiority and robustness in HLS performance prediction tasks.

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

## Dataset

## Baseline
- IRONMAN-PRO
- GCN+GF
- GAT+GF
- SAGE+GF
- RGCN+GF
