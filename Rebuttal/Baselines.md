# Additional Baseline Comparisons

To address the common concerns of reviewers unVk, 1Lxb, zr17's regarding more baselines comparisons, we conduct the following supplementary experiments.

## Baseline Comparisons

<img src="figure/Baseline.png">


### Result analysis
The supplementary baselines include advanced linear models like FreTS and TSMixer, Transformer-based models utilizing the Patch concept such as PatchTST and Crossformer, and advanced models resistant to non-stationarity like Koopa that employ Koopman operators. 

It can be observed that across several high-dynamic datasets in three scenarios, our designed MetaEformer still maintains superior performance (except Koopa in CBW). The dominance is attributed to the mechanism of **Meta Patterns Pooling** and **Echo**, which can effectively capture meta-patterns within dynamic loads and enhance model performance through series deconstruction.

### Notes
Regarding TimesNet [6], we encountered GPU RAM limitations during replication on 2* RTX 3090 and Tesla V100 setups. A downscaled TimesNet resulted in subpar performance, which would led to unfair comparisons. However, the models we have compared, including PatchTST, TSmixer, and FreTS, have shown stronger performance than TimesNet in the their reports, justifying the belief in MetaEformer's superiority compared to the TimesNet.

### To Reviewers
We appreciate the time you put into reviewing our work, and these latest relevant works are invaluable in demonstrating the performance of MetaEformer. We will make sure to include these results in the revised manuscript.

### References
[1] Yi K, Zhang Q, Fan W, et al. Frequency-domain MLPs are more effective learners in time series forecasting[J]. Advances in Neural Information Processing Systems, 2024, 36.

[2] Ekambaram V, Jati A, Nguyen N, et al. Tsmixer: Lightweight mlp-mixer model for multivariate time series forecasting[C] Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2023: 459-469.

[3] Nie Y, Nguyen N H, Sinthong P, et al. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers[C]//The Eleventh International Conference on Learning Representations. 2022.

[4] Zhang Y, Yan J. Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting[C]//The eleventh international conference on learning representations. 2022.

[5] Liu Y, Li C, Wang J, et al. Koopa: Learning non-stationary time series dynamics with koopman predictors[J]. Advances in Neural Information Processing Systems, 2024, 36.

[6] Wu H, Hu T, Liu Y, et al. Timesnet: Temporal 2d-variation modeling for general time series analysis[C]//The eleventh international conference on learning representations. 2022.
