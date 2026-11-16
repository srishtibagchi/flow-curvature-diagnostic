# Design Decisions

A record of project decisions and the reasoning behind them.

## Gaussian Baseline 

- Guidance: Use a simple Gaussian model as a baseline for comparison with more complex flow-based generative models.
- Decision: Implement a Gaussian baseline that captures the mean and covariance of the data distribution.
- Rationale: The Gaussian baseline provides a straightforward reference point to evaluate the performance of more sophisticated models. It helps in understanding the added value of flow-based approaches and ensures that any improvements are not due to trivial statistical properties of the data.

## Flow-Based Generative Models

- Guidance: Use flow-based generative models to capture complex data distributions beyond simple Gaussian assumptions.
- Decision: Implement normalizing flows that allow exact likelihood computation and flexible density modeling.
- Rationale: Flow-based models provide a powerful framework for modeling high-dimensional data with complex dependencies. They enable exact likelihood evaluation, which is useful for both training and evaluation, and offer a more expressive alternative to the Gaussian baseline.

## Evaluation Metrics

- Guidance: Use appropriate evaluation metrics to assess the performance of both Gaussian and flow-based generative models.
- Decision: Implement metrics such as log-likelihood, KL divergence, and sample quality measures to evaluate model performance.
- Rationale: Proper evaluation metrics are essential to quantify the effectiveness of the models. Log-likelihood and KL divergence provide insights into how well the models capture the data distribution, while sample quality measures help assess the realism and diversity of generated samples.

## Model Training

- Guidance: Train both Gaussian and flow-based generative models effectively while ensuring reproducibility and fair comparison.
- Decision: Use consistent training protocols, including data preprocessing, batch sizes, learning rates, and random seeds, across all models.
- Rationale: Consistent training protocols help ensure that performance differences are attributable to model capabilities rather than training discrepancies. Reproducibility is critical for validating results and comparing models fairly.

## Reproducibility and Randomness

- Guidance: Ensure that experiments are reproducible and that randomness is controlled across different runs.
- Decision: Set and document random seeds for all sources of randomness, including data shuffling, model initialization, and any stochastic training procedures.
- Rationale: Controlling randomness is crucial for reproducibility, allowing other researchers to replicate results and verify findings. Documenting the seeds ensures transparency and facilitates fair comparisons between different models and experimental setups.

## Data Preprocessing

- Guidance: Apply consistent data preprocessing steps to ensure fair comparison between models.
- Decision: Standardize input data by normalizing features to have zero mean and unit variance. Apply the same preprocessing pipeline to both training and evaluation datasets.
- Rationale: Consistent data preprocessing ensures that models are trained and evaluated on comparable inputs, reducing the risk of performance differences arising from data handling rather than model capabilities.

## Model Architecture

- Guidance: Design model architectures that are suitable for capturing the underlying data distribution while maintaining computational efficiency.
- Decision: Use architectures that balance expressiveness and computational cost, such as coupling layers in normalizing flows or structured covariance matrices in Gaussian models.
- Rationale: Choosing appropriate model architectures ensures that the models can effectively learn complex data distributions without incurring excessive computational overhead, facilitating both training and inference efficiency.

## Optimization and Training Procedures

- Guidance: Employ effective optimization algorithms and training procedures to ensure model convergence and performance.
- Decision: Use adaptive optimization algorithms such as Adam or RMSprop with carefully tuned learning rates and regularization techniques. Implement early stopping based on validation performance to prevent overfitting.
- Rationale: Effective optimization algorithms and training procedures help achieve stable convergence and improve model generalization. Early stopping prevents overfitting, ensuring that the models perform well on unseen data.

## Results Reporting

- Guidance: Report experimental results in a clear and comprehensive manner to facilitate comparison and interpretation.
- Decision: Present results using standardized metrics such as log-likelihood, KL divergence, and sample quality measures. Include both quantitative tables and qualitative visualizations where appropriate. Clearly indicate the experimental setup, including model configurations, training protocols, and random seeds.
- Rationale: Clear and comprehensive results reporting ensures that the findings are interpretable and reproducible. Standardized metrics and visualizations allow for meaningful comparisons between models, while detailed documentation of the experimental setup provides transparency and supports reproducibility.
