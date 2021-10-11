# Code Implementation

The State-space tensorization models along with compared benchmark ones introduced in the work _Entanglement-Structured LSTM Boosts Chaotic TimeSeries Forecasting_ are implemented and demonstrated in the [Mathematica notebook](https://github.com/owenyoung75/MERA-LSTM/blob/main/code/LongShortTermRestrictedMeraConeAllTasks.nb).

Besides, there is a set of [additional models](https://github.com/owenyoung75/MERA-LSTM/tree/main/code/tensorized_history) which tensorize a longer sequence of hidden/cell state variables implemented in TesnforFlow(v1.0).
This set of models are closely related to the tensorization idea we proposed in the manuscript, while taking advantage of different variables instead of cell states, which is more related to another [seminar work](http://www.stephanzheng.com/pdf/Yu_Zheng_Learning_Chaotic_Dynamics_using_Tensor_Recurrent_Neural_Networks_icml_2017.pdf) by R. Yu and etc. (ICML workshop, 2017), on higher order nonlinear dynamics. 
We have extended the original tensor-train RNN to more general tensorized structures. 

Further discussions from readers with interests would be welcomed.
