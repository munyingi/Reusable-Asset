"""
Model training and evaluation utilities.

This module contains functions for training machine learning models
and evaluating their performance.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    A class for training and evaluating machine learning models.
    """
    
    def __init__(self, model, task_type='classification'):
        """
        Initialize the ModelTrainer.
        
        Args:
            model: Scikit-learn compatible model
            task_type: Type of task ('classification' or 'regression')
        """
        self.model = model
        self.task_type = task_type
        self.is_trained = False
        
    def train(self, X_train, y_train):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info(f"Training {self.model.__class__.__name__}...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info("Training completed")
        
    def evaluate(self, X_test, y_test) -> dict:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.model.predict(X_test)
        
        if self.task_type == 'classification':
            metrics = self._evaluate_classification(y_test, y_pred, X_test)
        else:
            metrics = self._evaluate_regression(y_test, y_pred)
            
        logger.info("Evaluation completed")
        return metrics
    
    def _evaluate_classification(self, y_test, y_pred, X_test) -> dict:
        """
        Evaluate classification model.
        
        Args:
            y_test: True labels
            y_pred: Predicted labels
            X_test: Test features
            
        Returns:
            Dictionary of classification metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # Add ROC AUC if model supports predict_proba
        if hasattr(self.model, 'predict_proba'):
            try:
                y_pred_proba = self.model.predict_proba(X_test)
                if y_pred_proba.shape[1] == 2:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba, 
                                                      multi_class='ovr', average='weighted')
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {str(e)}")
        
        return metrics
    
    def _evaluate_regression(self, y_test, y_pred) -> dict:
        """
        Evaluate regression model.
        
        Args:
            y_test: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of regression metrics
        """
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2_score': r2_score(y_test, y_pred)
        }
        
        return metrics
    
    def cross_validate(self, X, y, cv=5) -> dict:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Labels
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary containing cross-validation scores
        """
        if self.task_type == 'classification':
            scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        else:
            scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        
        cv_results = {}
        for score in scoring:
            scores = cross_val_score(self.model, X, y, cv=cv, scoring=score)
            cv_results[score] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
        
        logger.info(f"Cross-validation completed with {cv} folds")
        return cv_results
    
    def hyperparameter_tuning(self, X_train, y_train, param_grid, cv=5) -> dict:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Dictionary of parameters to search
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary containing best parameters and scores
        """
        logger.info("Starting hyperparameter tuning...")
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=cv, 
            scoring='accuracy' if self.task_type == 'classification' else 'r2',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.is_trained = True
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Best parameters: {results['best_params']}")
        logger.info(f"Best score: {results['best_score']:.4f}")
        
        return results
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model
        """
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model


def compare_models(models: dict, X_train, y_train, X_test, y_test, task_type='classification') -> pd.DataFrame:
    """
    Compare multiple models and return their performance metrics.
    
    Args:
        models: Dictionary of model names and model objects
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        task_type: Type of task ('classification' or 'regression')
        
    Returns:
        DataFrame containing comparison of model performances
    """
    results = []
    
    for name, model in models.items():
        logger.info(f"Training and evaluating {name}...")
        
        trainer = ModelTrainer(model, task_type=task_type)
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
        
        metrics['model'] = name
        results.append(metrics)
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.set_index('model')
    
    logger.info("Model comparison completed")
    return comparison_df


if __name__ == "__main__":
    print("Model training module loaded successfully")
