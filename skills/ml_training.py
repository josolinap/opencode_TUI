#!/usr/bin/env python3
"""
ML Training Skill for Neo-Clone

Provides machine learning model training guidance and best practices.
"""

from base_skill import BaseSkill, SkillCategory, SkillResult
from typing import Dict, Any, List
import time


class MLTrainingSkill(BaseSkill):
    """Skill for machine learning model training guidance"""

    def __init__(self):
        super().__init__(
            name="ml_training",
            description="Machine learning model training guidance and best practices",
            category=SkillCategory.ML_TRAINING,
            capabilities=[
                "model_selection",
                "hyperparameter_tuning",
                "training_optimization",
                "performance_evaluation",
                "deployment_guidance"
            ]
        )

    def execute(self, params: Dict[str, Any]) -> SkillResult:
        """Execute ML training guidance"""
        start_time = time.time()

        try:
            task_type = params.get('task', 'general')
            model_type = params.get('model_type', 'unknown')
            dataset_info = params.get('dataset_info', {})

            if task_type == 'model_selection':
                result = self._provide_model_selection_guidance(model_type, dataset_info)
            elif task_type == 'hyperparameter_tuning':
                result = self._provide_hyperparameter_guidance(model_type)
            elif task_type == 'training_optimization':
                result = self._provide_training_optimization_tips()
            elif task_type == 'performance_evaluation':
                result = self._provide_evaluation_metrics_guidance()
            else:
                result = self._provide_general_ml_guidance()

            execution_time = time.time() - start_time

            return SkillResult(
                success=True,
                output=result,
                skill_name=self.name,
                execution_time=execution_time,
                metadata={
                    'task_type': task_type,
                    'model_type': model_type,
                    'guidance_provided': True
                }
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return SkillResult(
                success=False,
                output=f"ML training guidance failed: {str(e)}",
                skill_name=self.name,
                execution_time=execution_time,
                error_message=str(e)
            )

    def _provide_model_selection_guidance(self, model_type: str, dataset_info: Dict[str, Any]) -> str:
        """Provide model selection guidance"""
        guidance = []

        if model_type.lower() in ['classification', 'binary_classification']:
            guidance.append("For binary/multi-class classification:")
            guidance.append("• Logistic Regression - Simple, interpretable, good baseline")
            guidance.append("• Random Forest - Handles mixed data types, robust to outliers")
            guidance.append("• XGBoost - High performance, handles missing values")
            guidance.append("• Neural Networks - Complex patterns, requires more data")

        elif model_type.lower() in ['regression', 'prediction']:
            guidance.append("For regression tasks:")
            guidance.append("• Linear Regression - Simple, interpretable baseline")
            guidance.append("• Random Forest Regressor - Handles non-linear relationships")
            guidance.append("• Gradient Boosting (XGBoost/LightGBM) - High accuracy")
            guidance.append("• Neural Networks - Complex non-linear patterns")

        elif model_type.lower() in ['clustering', 'unsupervised']:
            guidance.append("For clustering/unsupervised learning:")
            guidance.append("• K-Means - Simple, fast, spherical clusters")
            guidance.append("• DBSCAN - Arbitrary shaped clusters, handles noise")
            guidance.append("• Gaussian Mixture Models - Probabilistic clustering")
            guidance.append("• Hierarchical Clustering - No need to specify k")

        else:
            guidance.append("General model selection considerations:")
            guidance.append("• Start with simple models (linear/logistic regression)")
            guidance.append("• Use cross-validation for model comparison")
            guidance.append("• Consider computational resources and training time")
            guidance.append("• Evaluate both accuracy and interpretability needs")

        return "\n".join(guidance)

    def _provide_hyperparameter_guidance(self, model_type: str) -> str:
        """Provide hyperparameter tuning guidance"""
        guidance = []

        guidance.append("Hyperparameter Tuning Strategy:")
        guidance.append("1. Start with default parameters")
        guidance.append("2. Use grid search or random search for small parameter spaces")
        guidance.append("3. Use Bayesian optimization for larger spaces")
        guidance.append("4. Always use cross-validation")
        guidance.append("")

        if 'neural' in model_type.lower() or 'network' in model_type.lower():
            guidance.append("Neural Network Hyperparameters:")
            guidance.append("• Learning Rate: 0.001-0.1 (start with 0.01)")
            guidance.append("• Batch Size: 32-256 (depends on memory)")
            guidance.append("• Hidden Layers: 1-3 (start with 1)")
            guidance.append("• Neurons per Layer: 64-512 (start with 128)")
            guidance.append("• Dropout: 0.2-0.5 (start with 0.2)")

        elif 'random_forest' in model_type.lower():
            guidance.append("Random Forest Hyperparameters:")
            guidance.append("• n_estimators: 100-1000 (start with 100)")
            guidance.append("• max_depth: 10-50 (start with None)")
            guidance.append("• min_samples_split: 2-10 (start with 2)")
            guidance.append("• min_samples_leaf: 1-5 (start with 1)")

        elif 'xgboost' in model_type.lower():
            guidance.append("XGBoost Hyperparameters:")
            guidance.append("• learning_rate (eta): 0.01-0.3 (start with 0.1)")
            guidance.append("• max_depth: 3-10 (start with 6)")
            guidance.append("• n_estimators: 100-1000 (start with 100)")
            guidance.append("• subsample: 0.8-1.0 (start with 0.8)")

        return "\n".join(guidance)

    def _provide_training_optimization_tips(self) -> str:
        """Provide training optimization tips"""
        tips = [
            "Training Optimization Tips:",
            "",
            "1. Data Preprocessing:",
            "   • Handle missing values appropriately",
            "   • Normalize/standardize features",
            "   • Remove outliers if appropriate",
            "   • Encode categorical variables",
            "",
            "2. Training Best Practices:",
            "   • Use early stopping to prevent overfitting",
            "   • Implement learning rate scheduling",
            "   • Use data augmentation if applicable",
            "   • Monitor validation loss during training",
            "",
            "3. Performance Optimization:",
            "   • Use appropriate batch sizes",
            "   • Implement gradient clipping for stability",
            "   • Use mixed precision training if supported",
            "   • Consider distributed training for large datasets",
            "",
            "4. Debugging Training:",
            "   • Check for data leakage",
            "   • Monitor learning curves",
            "   • Validate assumptions about data",
            "   • Use proper train/validation/test splits"
        ]

        return "\n".join(tips)

    def _provide_evaluation_metrics_guidance(self) -> str:
        """Provide evaluation metrics guidance"""
        guidance = [
            "Model Evaluation Metrics:",
            "",
            "Classification Metrics:",
            "• Accuracy: Overall correctness (not suitable for imbalanced data)",
            "• Precision: True positives / (True positives + False positives)",
            "• Recall: True positives / (True positives + False negatives)",
            "• F1-Score: Harmonic mean of precision and recall",
            "• AUC-ROC: Area under ROC curve (probability ranking)",
            "",
            "Regression Metrics:",
            "• MAE: Mean Absolute Error (interpretable, robust to outliers)",
            "• MSE: Mean Squared Error (penalizes large errors)",
            "• RMSE: Root Mean Squared Error (same units as target)",
            "• R²: Coefficient of determination (explained variance)",
            "",
            "General Best Practices:",
            "• Use cross-validation for reliable estimates",
            "• Evaluate on held-out test set",
            "• Consider business metrics, not just technical metrics",
            "• Use confusion matrices for classification insights",
            "• Plot learning curves to diagnose issues"
        ]

        return "\n".join(guidance)

    def _provide_general_ml_guidance(self) -> str:
        """Provide general ML guidance"""
        guidance = [
            "General Machine Learning Workflow:",
            "",
            "1. Problem Definition:",
            "   • Clearly define the problem and success criteria",
            "   • Understand the business context",
            "   • Identify available data sources",
            "",
            "2. Data Preparation:",
            "   • Exploratory data analysis (EDA)",
            "   • Feature engineering and selection",
            "   • Data cleaning and preprocessing",
            "   • Train/validation/test splits",
            "",
            "3. Model Development:",
            "   • Start with simple models as baselines",
            "   • Use cross-validation for model selection",
            "   • Hyperparameter tuning with proper validation",
            "   • Ensemble methods for improved performance",
            "",
            "4. Model Deployment:",
            "   • Model serialization and versioning",
            "   • Performance monitoring in production",
            "   • A/B testing for model improvements",
            "   • Continuous learning and retraining"
        ]

        return "\n".join(guidance)
