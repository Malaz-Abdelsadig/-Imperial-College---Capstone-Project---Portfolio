
# Model Card

See the [example Google model cards](https://modelcards.withgoogle.com/model-reports)
for inspiration. 

## Model Description

**Input:**
FTIR spectra data, focusing on the wavenumber range of 1500-1800 cm⁻¹.
Parameters for the absorption band detection algorithm, such as absorption intensities (%) and wavenumbers (cm-1)

**Output:**
Identification of significant absorption bands corresponding to (C=O) of carboxylic group.
Prediction of pH sensitivity based on the presence and characteristics of these absorption bands.

**Model Architecture:**
Absorption Band Detection Algorithm: Identifies significant absorption bands in the FTIR spectrum.
Random Forest Model: An ensemble learning method consisting of multiple decision trees to analyse the FTIR spectra data. This model is robust and capable of handling complex, non-linear relationships within the data.
Bayesian Optimization: Utilized to fine-tune the parameters of both the absorption band detection algorithm and the Random Forest model, ensuring optimal performance.

## Performance

**Metrics:**
Accuracy: 0.8596 (approximately 86% of samples correctly classified)
Precision: 0.8862 (88.62% of predicted pH-responsive samples are correct)
Recall: 0.8912 (89.12% of actual pH-responsive samples correctly identified)
F1 Score: 0.8887 (balance between precision and recall)

**Data:**
FTIR spectra data from various bio/polymer samples.
Performance evaluated on a validation set separate from the training data.

**Insights:**
1.	The model shows a strong ability to correctly classify both inert and pH-responsive samples, though some misclassifications remain.
2.	High accuracy and F1 score indicate effective balance between precision and recall.
3.	Slightly higher recall suggests better identification of pH-responsive samples, which is beneficial for this application.
4.	Further reduction of false positives and negatives could enhance reliability, potentially through additional feature engineering, more data, or further fine-tuning.
   
## Limitations

1.	Data Dependency: The model’s performance is highly dependent on the quality and quantity of the FTIR spectra data.
2.	Computational Resources: Bayesian optimization can be computationally intensive, requiring significant resources for large datasets.
3.	Spectral Quality: The quality of FTIR spectra can be affected by spectral resolution and noise, impacting the accuracy of absorption band detection.
4.	Parameter Sensitivity: The performance of the absorption band detection algorithm is highly dependent on the chosen parameters, with a risk of suboptimal settings affecting detection accuracy.
5.	Model Complexity: While the Random Forest model is robust, it may not capture all intricate relationships within the data. Integrating more advanced techniques like deep learning could address this but adds complexity and computational requirements.
6.	Physicochemical Properties: The current model does not fully integrate other relevant physicochemical properties, requiring significant dataset.
   
## Trade-offs

1.	Performance vs. Computational Cost: Bayesian optimization improves model performance but increases computational cost and time.
2.	Sensitivity vs. Specificity: There may be trade-offs between sensitivity (detecting all possible C=O groups) and specificity (avoiding false positives).
3.	Parameter Tuning: The iterative nature of Bayesian optimization means that finding the optimal parameters can be time-consuming, especially for large parameter spaces.
4.	Scalability: Exploring the scalability of this approach for industrial applications may require additional adjustments and validations.

