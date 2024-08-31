# KDD2024 Rebuttal Openreview提交版

# Reviewer 1

**Comment**

First of all, thanks for your positive feedback on the novelty and technical contribution of this work. We provide the responses to your main concerns as follows.

> ***Q1: Training Speed or Computational Efficiency***

Thank you for highlighting the importance of computational efficiency. We acknowledge the concern regarding the complexity added by the MetaEformer. A detailed comparison of the training times is available at [1].

Furthermore, we analyze the computational complexity introduced by each component of MetaEformer and contrasted it with the essential components \(such as Self-Attention and FeedForward layers\) of Transformer-like models. Due to space constraints, we provide the complete computational breakdown at the above link.

The conclusion are as follows:

The complexity of the native components of Transformer-like models exceeds the new components of MetaEformer by an order of $\frac{Ld+d^2}{K}$times. With common deep learning task settings, this scale can be hundreds of times larger. Therefore, we assert that the additional computational complexity from MetaEformer is negligible in contrast to the native complexity of Transformer-like models. The experimental results also demonstrate a significant increase in the model's accuracy and robustness for complex dynamic system load forecasting with minimal extra computational cost.

We hope this explanation clarifies your concern will make sure to provide more details in the revised manuscript.

> ***Q2: Broader Range of Baselines***

We thank you for the additional SOTA models suggested. We have made every effort to compare the work in the areas of interest to our paper \(decomposition-based, clustering-based, and anti-non-stationary\), providing up to eight of the latest baselines. However, some models were not included in our comparisons due to their divergence from the real-world system load scenarios or their methodologies not clearly falling within the three focused directions.

To address your concern and compare MetaEformer to a broader range of algorithms, we include the three additional works you referenced in our analysis. The results and discussion of this analysis can be found at [2].

We are thankful for your valuable feedback and will make sure to include the above disscussion in our revised version appropriately.

[1] anonymous.4open.science/r/MetaEformer-4313/Rebuttal/Complexity\_Analysis.md

[2] anonymous.4open.science/r/MetaEformer-4313/Rebuttal/Baselines.md.

‍

# Reviewer 2

**Comment**

Thank you for your positive feedback on the application and the scenario generalization of the approach. To address your concerns:

> ***Q1: Technical Novelty***

We appreciate your concern regarding novelty. To clarify: the decomposition, clustering, and adaptation concepts are widely adopted ideas in forecasting, and the distinctiveness of our approach lies in how these ideas are implemented and the role they play, which we summarize as follows:

1. **Why:**     Our method is specifically designed to tackle the complexity and dynamics challenges, informed by Wavelet Transform principles, we employ decomposition and clustering to extract and purify periodic patterns into meta-patterns. To the best of our knowledge, we are the first to focus on more representative meta-patterns as a way to address a wide range of forecasting tasks.
2. **How:**     We devise a novel clustering mechanism in Meta-pattern Pooling, which includes a unique similarity measure, threshold-based selection, and weighted generation of meta-patterns, offering lower complexity and real-time update capabilities suitable for time series data. The accompanying Echo Layer is also a novel mechanism to deconstruct load series and reconstruct patterns by skillfully utilizing meta-patterns. In addition, we design SI Embedding and Echo Padding mechanisms that are relevant to the scenarios, which are straightforward but effective.
3. **Benefits:**     MetaEformer demonstrates superior performance across multiple system scenarios, achieving substantial improvements over the SOTA models.

We hope this explanation clarifies your concern and will provide more details in the revised manuscript.

> ***Q2: Comparison to Recent Works***

We appreciate the suggested recent works and conduct additional comparisons. Results are reported in [1].

> ***Q3: Related Works on Distribution Shift***

Thanks for your valuable suggestion. Concept-drift, central to our study, is akin to distribution shift (cited as [25] in the mansucript). Besides, many referenced works in cluster-based and anti-nonstationary sections directly address distribution shifts, including [14,15,31,32] and [22-24]. We also articulate how these works tackles dynamic challenges like distribution shifts ($\S2$, the 5th and the penultimate paragraph).

We will enhance the discussion and integrate additional works as per your recommendations in the revised version.

[1] anonymous.4open.science/r/MetaEformer-4313/Rebuttal/Baselines.md.

‍

# Reviewer 3

**Comment**

First of all, thanks for your positive feedback on the novelty of the proposal and the comprehensiveness of the experiments. We provide the responses to your main concerns as follows.

> ***Q1: Complexity Analysis***

Thank you for your valuable suggestion. We recognize the importance of complexity analysis and computational efficiency in comparison to SOTA models. We have documented a detailed training time comparison and complexity analysis in [1]. Rest assured, we will include a comprehensive complexity analysis in the revised version of our paper to elucidate the computational efficiency of our model.

> ***Q2: Impact of K on Model Complexity***

Thank you for your insightful question. The Echo Layer's complexity is indeed proportional to $K$ with $O(BsK)$. While increasing $K$  does raise the complexity of the Echo Layer. However, as illustrated in <u>link,</u> the additional introduced computational complexity of MetaEformer is a fraction—specifically $\frac{K}{Ld+d^2}$ of the complexity incurred by the self-attention and FeedForward computations in transformer-like models. As $K$ is not typically large, this ensures the added complexity is several orders of magnitude less significant, rendering it practically negligible.

We appreciate this opportunity to clarify these aspects and are committed to enhancing the manuscript to improve clarity and understanding.

[1] anonymous.4open.science/r/MetaEformer-4313/Rebuttal/Complexity\_Analysis.md

‍

# Reviewer 4

**Comment**

Thanks for acknowledging the novelty and comprehensiveness of our work. To address your concerns:

> ***Q1: Sim for Different Frequencies***

Despite theoretical concerns, MeteEformer's implementation ensures that Sim correctly plays the role of similarity metrics and the extraction of meta-patterns.:

1. **Slicing:**  As the calculation unit, waveform's length is deliberately short after slicing, which reduces potential series sub-periods. This granularity ensures $Sim$'s focus on shape effectively distinguishes factors like frequency or phase.
2. **MPP Diversity:**  The construction and update of the MPP introduce varied meta-patterns, ensuring comprehensive coverage even with some redundancy.
3. **Echo Mechanism:**  Echo adaptively selects Top-K meta-patterns for time series deconstruction, accommodating MPP redundancy.

Further details will be added in the revision.

> ***Q2: SI Embedding***

We conduct ablation study to show its impact. Please refer to [1].

> ***Q3: Multi-Variate Interaction***

While our initial approach focused on single-variate forecasts, we recognize the significance of handling multiple variables.

The Channel-Independence where each channel contains a single univariate time series that shares the same embedding and Transformer weights across all the series, demonstrated in models like PatchTST and TSMixer, could seamlessly extend MetaEformer to multi-variate forecasting. This process also involves initial MPP on one variable, then integrating other variables to update the MPP.

We will add more discussion in the revised version and try to experimentally demonstrate the effectiveness of this scheme.

> ***Q4: Applicability and Transfer Learning***

MPP is not shared across datasets and variables, it is designed dataset-specific to maintain fairness in evaluation. While MetaEformer introduces additional complexity, as detailed in added Complexity Analysis in rebuttal, it is negligible compared to the transformer's original computations.

The use of meta-patterns for transfer learning holds great potential, as many scenarios exhibit similar load patterns. This could streamline model efficiency and reduce cold starts, potentially allowing MetaEformer to adapt pre-accumulated meta-patterns for new scenarios with minimal updates.

We will provid a deeper analysis of the cross-domain applicability and validate this potential through further experiments.

[1] anonymous.4open.science/r/MetaEformer-4313/Rebuttal/Ablation.md

‍

# Reviewer 5

**Comment**

First of all, thanks for your positive feedback on the novelty, the presenation quality and reproducibility of this work. We provide the responses to your main concerns as follows.

> ***Q1:***   ***Explanation on &quot;alpha&quot; Parameter***

Thanks for your valuable suggestion. To clarify, the $\alpha$ parameter and the MPP size $P$ both influence the threshold $\tau$, controlling the addition of $\sigma(SM)$ to adjust $\tau$.

However, $P$ plays a more pivotal role in MPP construction—larger $P$ implies a larger $\tau$, allowing for more diverse meta-patterns to populate the MPP, and $\alpha$ is only used to scale $P$ to logically set $\tau$. This rationale behind fixing $\alpha$ while adjusting $P$ aligns with the scenario's complexity, as demonstrated in $\S$5.4.

We hope this explanation clarifies your concern, and we will make sure to provide more details in the revised manuscript.

> ***Q2:***   ***Execution Time Comparison***

Thank you for your valuable suggestion. We have conducted an extensive analysis against the reported baselines, highlighting MetaEformer's training and inference efficiency. Please refer to [1].

> ***Q3: High-value Slices and Similarity***

1. All series undergo standardization, minimizing absolute value disparities among slices and ensuring high-value data does not disproportionately affect similarity calculations. Actually, the standardization is introduced to equalize standard deviations in that case.
2. The primary utility of $Sim$ is to identify slices most similar to a given slice $\mathcal{W}_i$, with comparisons conducted on a uniform scale, ensuring fairness and validity.
3. For slice $\mathcal{W}_i$ with extreme values, the penalization mechanism ensures its limited use in MPP construction and updating, mitigating undue influence by restricting its participation to a single meta-pattern creation (accomplished by maintaining an array of used slice). Furthermore, the Echo mechanism's selection of Top-K meta-patterns and continuous MPP updates dilute any significant impact from $\mathcal{W}_i$, rendering it negligible.

We will surely enhance our manuscript based on your insightful feedback.

[1] anonymous.4open.science/r/MetaEformer-4313/Rebuttal/Complexity\_Analysis.md

‍

# Reviewer 6

**Comment**

Thank you for recognizing the importance and novelty of our work. To address your concerns:

> ***Q1: Limitations of existing methods***

Fig. 2 illustrates the capabilities of various approaches, emphasizing the combined advantages rather than suggesting limitations. The primary goal is to highlight how MetaEformer ingeniously integrates these strengths to address the system load forecasting.

The detailed discussion on the limitations of existing work is in Introduction (Paragraph 7) and the Related Works section. To summarize:

Decomposition-based methods falter in adapting to the dynamic nature of system loads, particularly with new or transitioning entities. Clustering-based strategies suffer from high computational costs and potential information leakage. Anti-non-stationary methods improve forecasting for dynamic series but risk losing essential dynamics and lack clear interpretability regarding system dynamics representation ($\S 2$, last paragraph).

We'll ensure the revised manuscript offers a concise explanation.

> ***Q2: Exclusion of Trend Components***

The decision to ignore trend components is twofold:

1. Seasonal components, being more intricate and representational, are better suited for meta-pattern extraction to reconstruct new patterns effectively.
2. Early model iterations included trend components, but their contribution was minimal compared to the computational overhead they introduced.

Your question inspires us to explore alternative ways to incorporate trend information. Further exploration on this will be considered.

> ***Q3: Koopman-based Methods***

We appreciate the suggested methods and conduct comparisons with the models you kindly provided. Please refer to [1].

> ***Q4: Sensitivity of Slice Size***

We conduct an additional analysis against slice size. Please refer to [2].

> ***Q5: Model Interpretability***

MetaEformer's interpretability lies in explaining the complexity and dynamics of the system and why the model performs well, which is critical for operators in real industrial systems.

MetaEformer reflects the scenario's complexity (larger MPP size) and dynamics (frequent introduction of new meta-patterns). Moreover, the Echo Layer provides detailed insights into the model's predictions with explicit weights.

Finally, for your valuable suggestions about visualization results and spelling error, we will make sure to improve them in the revised version.

[1] anonymous.4open.science/r/MetaEformer-4313/Rebuttal/Baselines.md.

[2] anonymous.4open.science/r/MetaEformer-4313/Rebuttal/Sensitivity.md

‍
