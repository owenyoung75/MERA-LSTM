## TensorFlow Implementation of general tensor-network models in sequencial modeling

This part of code is based on Yu's [original repository](https://github.com/yuqirose/TensorRNN) for the work on higher-order nonlinear dynamics[<sup>1</sup>](http://www.stephanzheng.com/pdf/Yu_Zheng_Learning_Chaotic_Dynamics_using_Tensor_Recurrent_Neural_Networks_icml_2017.pdf). 
We have extended the original tensor-train RNN model int 3 directions:
1. tensorization in either hidden state or cell state;
2. tensorizaed representation in either polynomial space or timelag-lattice space;
3. general tensor-network structure beyond the simplest one: Matrix-Product-State (i.e. tensor train model).

### Model type:
- Recerrent-structure-based models, using hidden variables;
- State-variable-based models.

### Representation type:
- Polynomial-lattice representation, single-site-dimension = HL + 1
- Timelag-lattice representation, single-site-dimension = HP + 1

#### References:
1. _[Learning Chaotic Dynamics using Tensor Recurrent Neural Networks]((http://www.stephanzheng.com/pdf/Yu_Zheng_Learning_Chaotic_Dynamics_using_Tensor_Recurrent_Neural_Networks_icml_2017.pdf)). Rose Yu, Stephan Zheng, and Yan Liu, Proceedings of the ICML 17 Workshop on Deep Structured Prediction, Sydney, Australia, PMLR 70, 2017_
