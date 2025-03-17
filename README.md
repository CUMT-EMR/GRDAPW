# GRDAPW
本仓库包含论文"**Adaptive data preprocessing and multi-indicator fusion early warning for gas and roof disaster risks in coal mines**"中所有代码，其代码层级如下：

1. Outlier Handling Model

其中包含基于xLSTM的半监督协同训练的模型代码（注意事项：需要先安装xLSTM包，请参考仓库[NX-AI/xlstm: Official repository of the xLSTM.](https://github.com/NX-AI/xlstm)）

2. Noise Reduction Model

其中包含基于PSO-小波阈值去噪-OPTICS降噪模型的代码，提供了结合AE、EMR信号的降噪实例。

3. BayOTIDE在线实时插补模型

由于仅在系统应用中使用了该模型，该模型代码来源仓库[xuangu-fang/BayOTIDE: BayOTIDE-Bayesian Online Multivariate Time Series Imputation with Functional Decomposition (ICML 2024 spotlight)](https://github.com/xuangu-fang/BayOTIDE)，我们在此基础上进一步开发了符合我们数据分布规律的数据预处理代码、配置文件，生成了瓦斯、顶板预警相关指标的数据集合，具体文件在文件夹中上传。

4. MTAFM瓦斯、顶板融合预警模型

我们提供了该模型的结构代码，可以非常方便利用其的构建自己的任务。
