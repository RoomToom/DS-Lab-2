"""
Task 2: Outlier Detection and Treatment with Advanced Visualization

This module implements various outlier detection and treatment techniques,
including statistical methods, machine learning-based detection, and 
comprehensive visualization of outliers and their impact.

Author: MLDS Course
Date: December 2025
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from scipy import stats
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')


class OutlierAnalyzer:
    """
    A comprehensive outlier analyzer that detects and treats outliers using
    multiple methods and provides detailed visualizations.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the analyzer with a dataset.
        
        Parameters:
            data (pd.DataFrame): Input DataFrame to analyze
        """
        self.data = data.copy()
        self.original_data = data.copy()
        self.outlier_masks = {}
        self.treated_datasets = {}
        
    def detect_outliers_zscore(self, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers using Z-score method.
        
        Parameters:
            threshold (float): Z-score threshold for outlier detection
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        if threshold <= 0:
            raise ValueError('Threshold must be positive.')
        data = self.data.select_dtypes(include=np.number)
        mask = ((data - data.mean()) / data.std(ddof=0).replace(0, np.nan)).abs() > threshold
        self.outlier_masks['zscore'] = mask
        return mask
    
    def detect_outliers_iqr(self, multiplier: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Parameters:
            multiplier (float): IQR multiplier for boundary calculation
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        if multiplier <= 0:
            raise ValueError('Multiplier must be positive.')
        data = self.data.select_dtypes(include=np.number)
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        mask = (data < q1 - multiplier * (q3 - q1)) | (data > q3 + multiplier * (q3 - q1))
        self.outlier_masks['iqr'] = mask
        return mask
    
    def detect_outliers_isolation_forest(self, contamination: float = 0.1, 
                                        random_state: int = 42) -> pd.DataFrame:
        """
        Detect outliers using Isolation Forest algorithm.
        
        Parameters:
            contamination (float): Expected proportion of outliers in the dataset
            random_state (int): Random seed for reproducibility
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        data = self.data.select_dtypes(include=np.number)
        labels = IsolationForest(contamination=contamination, random_state=random_state).fit_predict(data.fillna(data.median()).fillna(0))
        mask = pd.DataFrame({'outlier': labels == -1}, index=data.index)
        self.outlier_masks['isolation_forest'] = mask
        return mask
    
    def detect_outliers_lof(self, n_neighbors: int = 20, 
                           contamination: float = 0.1) -> pd.DataFrame:
        """
        Detect outliers using Local Outlier Factor (LOF) algorithm.
        
        Parameters:
            n_neighbors (int): Number of neighbors to consider
            contamination (float): Expected proportion of outliers
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        data = self.data.select_dtypes(include=np.number)
        labels = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination).fit_predict(data.fillna(data.median()).fillna(0))
        mask = pd.DataFrame({'outlier': labels == -1}, index=data.index)
        self.outlier_masks['lof'] = mask
        return mask
    
    def get_outlier_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of outliers detected by different methods.
        
        Returns:
            DataFrame with outlier counts for each method
        """
        return pd.DataFrame({name: mask.sum() for name, mask in self.outlier_masks.items()}).fillna(0).astype(int)
    
    def remove_outliers(self, method: str = 'iqr') -> pd.DataFrame:
        """
        Remove outliers from the dataset.
        
        Parameters:
            method (str): Detection method to use
            
        Returns:
            pd.DataFrame: Dataset with outliers removed
        """
        if method not in self.outlier_masks:
            raise ValueError('Outlier mask not found: ' + method)
        data = self.data.loc[~self.outlier_masks[method].any(axis=1)].copy()
        self.treated_datasets['removed_' + method] = data
        return data
    
    def cap_outliers(self, method: str = 'iqr', multiplier: float = 1.5) -> pd.DataFrame:
        """
        Cap outliers using winsorization.
        
        Parameters:
            method (str): Detection method to use
            multiplier (float): IQR multiplier for bounds
            
        Returns:
            pd.DataFrame: Dataset with capped outliers
        """
        if method not in ('iqr', 'zscore') or multiplier <= 0:
            raise ValueError('Use iqr or zscore with a positive multiplier.')
        data = self.data.copy()
        columns = data.select_dtypes(include=np.number).columns
        values = data[columns]
        if method == 'iqr':
            q1, q3 = values.quantile(0.25), values.quantile(0.75)
            lower, upper = q1 - multiplier * (q3 - q1), q3 + multiplier * (q3 - q1)
        else:
            lower, upper = values.mean() - multiplier * values.std(), values.mean() + multiplier * values.std()
        data[columns] = values.clip(lower, upper, axis=1)
        self.treated_datasets['capped'] = data
        return data
    
    def transform_outliers_log(self) -> pd.DataFrame:
        """
        Transform data using log transformation.
        
        Returns:
            pd.DataFrame: Dataset with log-transformed values
        """
        data = self.data.copy()
        columns = data.select_dtypes(include=np.number).columns
        values = data[columns]
        data[columns] = np.log1p(values - values.min().clip(upper=0))
        self.treated_datasets['log_transform'] = data
        return data
    
    def transform_outliers_boxcox(self) -> pd.DataFrame:
        """
        Transform data using Box-Cox/Yeo-Johnson transformation.
        
        Returns:
            pd.DataFrame: Dataset with power-transformed values
        """
        data = self.data.copy()
        columns = [col for col in data.select_dtypes(include=np.number) if data[col].nunique() > 1]
        if columns:
            data[columns] = PowerTransformer(method='yeo-johnson').fit_transform(data[columns])
        self.treated_datasets['boxcox_transform'] = data
        return data
    
    def evaluate_model_performance(self, target_col: str, 
                                   dataset_name: str = 'original') -> Dict[str, float]:
        """
        Evaluate model performance with and without outliers.
        
        Args:
            target_col: Name of target column
            dataset_name: Dataset to evaluate ('original' or treated dataset name)
            
        Returns:
            Dictionary with performance metrics
        """
        from sklearn.model_selection import KFold
        from sklearn.metrics import r2_score
        if dataset_name != 'original' and dataset_name not in self.treated_datasets:
            raise ValueError('Dataset not found: ' + dataset_name)
        X = self.original_data.drop(columns=target_col).select_dtypes(include=np.number)
        y = self.original_data[target_col]
        scores = []
        # Межі викидів і перетворення оцінюються тільки на тренувальній частині.
        for train, test in KFold(5, shuffle=True, random_state=42).split(X):
            a, b, target = X.iloc[train].copy(), X.iloc[test].copy(), y.iloc[train]
            if dataset_name.startswith('removed_'):
                analyzer = OutlierAnalyzer(a)
                method = dataset_name.removeprefix('removed_')
                detector = getattr(analyzer, 'detect_outliers_' + method)
                keep = ~detector().any(axis=1)
                a, target = a.loc[keep], target.loc[keep]
            elif dataset_name == 'capped':
                q1, q3 = a.quantile(.25), a.quantile(.75)
                lower, upper = q1 - 1.5 * (q3-q1), q3 + 1.5 * (q3-q1)
                a, b = a.clip(lower, upper, axis=1), b.clip(lower, upper, axis=1)
            elif dataset_name == 'log_transform':
                shift = a.min().clip(upper=0)
                a, b = np.log1p(a-shift), np.log1p((b-shift).clip(lower=0))
            elif dataset_name == 'boxcox_transform':
                transformer = PowerTransformer()
                a, b = transformer.fit_transform(a), transformer.transform(b)
            model = LinearRegression().fit(a, target)
            scores.append(r2_score(y.iloc[test], model.predict(b)))
        return {'r2_score': float(np.mean(scores)), 'r2_std': float(np.std(scores)), 'mean_cv_score': float(np.mean(scores))}
    
    def visualize_outliers_boxplot(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create box plots showing outliers.
        
        Parameters:
            figsize (Tuple[int, int]): Figure size for plots
            
        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        columns = self.data.select_dtypes(include=np.number).columns[:6]
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        for ax, col in zip(axes.flat, columns):
            values = self.data[col].dropna()
            ax.boxplot(values)
            ax.set_title(col)
        for ax in list(axes.flat)[len(columns):]:
            ax.set_visible(False)
        fig.tight_layout()
        return fig
    
    def visualize_outliers_violin(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create violin plots showing distribution with outliers.
        
        Args:
            figsize: Figure size for plots
        """
        columns = self.data.select_dtypes(include=np.number).columns[:6]
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        for ax, col in zip(axes.flat, columns):
            values = self.data[col].dropna()
            ax.violinplot(values, showmeans=True, showmedians=True)
            ax.set_title(col)
        for ax in list(axes.flat)[len(columns):]:
            ax.set_visible(False)
        fig.tight_layout()
        return fig
    
    def visualize_outliers_scatter(self, x_col: str, y_col: str, 
                                   figsize: Tuple[int, int] = (15, 5)):
        """
        Create scatter plots showing outliers detected by different methods.
        
        Args:
            x_col: Column for x-axis
            y_col: Column for y-axis
            figsize: Figure size for plots
        """
        if x_col not in self.data or y_col not in self.data:
            print('Column not found')
            return None
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        for ax, method in zip(axes, ['zscore', 'iqr', 'isolation_forest']):
            if method not in self.outlier_masks:
                getattr(self, 'detect_outliers_' + method)()
            outliers = self.outlier_masks[method].any(axis=1)
            for mask, color, label in [(~outliers, 'steelblue', 'Normal'), (outliers, 'tomato', 'Outlier')]:
                ax.scatter(self.data.loc[mask, x_col], self.data.loc[mask, y_col], c=color, label=label, alpha=.7)
            ax.set(xlabel=x_col, ylabel=y_col, title=method)
            ax.legend()
        fig.tight_layout()
        return fig
    
    def visualize_3d_outliers(self, x_col: str, y_col: str, z_col: str, 
                             method: str = 'isolation_forest',
                             figsize: Tuple[int, int] = (12, 8)):
        """
        Create three-dimensional scatter plot showing multivariate outliers.
        
        Args:
            x_col, y_col, z_col: Columns for three-dimensional axes
            method: Detection method to use
            figsize: Figure size for plot
        """
        if method not in self.outlier_masks or any(col not in self.data for col in [x_col, y_col, z_col]):
            print('Columns or detection method not found')
            return None
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        outliers = self.outlier_masks[method].any(axis=1)
        for mask, color, label in [(~outliers, 'steelblue', 'Normal'), (outliers, 'tomato', 'Outlier')]:
            ax.scatter(self.data.loc[mask, x_col], self.data.loc[mask, y_col], self.data.loc[mask, z_col], c=color, label=label, alpha=.7)
        ax.set(xlabel=x_col, ylabel=y_col, zlabel=z_col, title=method)
        ax.legend()
        fig.tight_layout()
        return fig
    
    def visualize_treatment_comparison(self, column: str, 
                                      figsize: Tuple[int, int] = (15, 10)):
        """
        Compare distributions before and after different treatments.
        
        Args:
            column: Column to visualize
            figsize: Figure size for plots
        """
        if column not in self.data.columns:
            print(f"Column '{column}' not found")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        # Original data
        ax = axes[0]
        self.original_data[column].hist(bins=30, ax=ax, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(self.original_data[column].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax.set_title('Original Data', fontweight='bold')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Treated datasets
        treatment_names = list(self.treated_datasets.keys())[:5]
        for idx, name in enumerate(treatment_names, start=1):
            ax = axes[idx]
            treated_data = self.treated_datasets[name]
            if column in treated_data.columns:
                treated_data[column].hist(bins=30, ax=ax, color='lightcoral', 
                                        edgecolor='black', alpha=0.7)
                ax.axvline(treated_data[column].mean(), color='red', 
                          linestyle='--', linewidth=2, label='Mean')
                ax.set_title(f'{name.replace("_", " ").title()}', fontweight='bold')
                ax.set_xlabel(column)
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(treatment_names) + 1, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Treatment Comparison: {column}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def visualize_qq_plots(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create Q-Q plots to assess normality before and after treatment.
        
        Args:
            figsize: Figure size for plots
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:6]
        
        if len(numeric_cols) == 0:
            print("No numeric columns to visualize")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            
            stats.probplot(self.data[col].dropna(), dist="norm", plot=ax)
            ax.set_title(f'Q-Q Plot: {col}', fontweight='bold')
            ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Q-Q Plots for Normality Assessment', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig


def create_sample_dataset_with_outliers() -> pd.DataFrame:
    """
    Create a sample dataset with outliers for demonstration.
    
    Returns:
        pd.DataFrame: Dataset with outliers
    """
    np.random.seed(42)
    n_samples = 300
    
    # Create base data
    data = {
        'feature1': np.random.normal(100, 15, n_samples),
        'feature2': np.random.exponential(50, n_samples),
        'feature3': np.random.normal(500, 100, n_samples),
        'feature4': np.random.uniform(0, 100, n_samples),
        'target': np.random.normal(200, 30, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add synthetic target relationship
    df['target'] = 0.5 * df['feature1'] + 0.3 * df['feature2'] + np.random.normal(0, 10, n_samples)
    
    # Inject outliers
    # Extreme high values in feature1
    outlier_indices = np.random.choice(df.index, size=15, replace=False)
    df.loc[outlier_indices, 'feature1'] = np.random.uniform(200, 300, len(outlier_indices))
    
    # Extreme low values in feature2
    outlier_indices = np.random.choice(df.index, size=10, replace=False)
    df.loc[outlier_indices, 'feature2'] = np.random.uniform(200, 400, len(outlier_indices))
    
    # Random extreme values in feature3
    outlier_indices = np.random.choice(df.index, size=20, replace=False)
    # Split outliers into low and high
    low_outliers = outlier_indices[:len(outlier_indices)//2]
    high_outliers = outlier_indices[len(outlier_indices)//2:]
    df.loc[low_outliers, 'feature3'] = np.random.uniform(0, 100, len(low_outliers))
    df.loc[high_outliers, 'feature3'] = np.random.uniform(1000, 1500, len(high_outliers))
    
    return df


def main():
    """
    Main function demonstrating the complete Task 2 workflow.
    
    Returns:
        OutlierAnalyzer: Configured analyzer object
    """
    print("=" * 80)
    print("Task 2: Outlier Detection and Treatment with Advanced Visualization")
    print("=" * 80)
    print()
    
    # Create sample dataset
    print("Creating sample dataset with outliers...")
    df = create_sample_dataset_with_outliers()
    print(f"Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
    print()
    
    # Initialize analyzer
    analyzer = OutlierAnalyzer(df)
    
    # Detect outliers using different methods
    print("1. OUTLIER DETECTION")
    print("-" * 80)
    
    print("  a) Z-Score Detection...")
    zscore_outliers = analyzer.detect_outliers_zscore(threshold=3.0)
    print(f"     Outliers detected: {zscore_outliers.sum().sum()}")
    
    print("  b) IQR Detection...")
    iqr_outliers = analyzer.detect_outliers_iqr(multiplier=1.5)
    print(f"     Outliers detected: {iqr_outliers.sum().sum()}")
    
    print("  c) Isolation Forest Detection...")
    iso_outliers = analyzer.detect_outliers_isolation_forest(contamination=0.1)
    print(f"     Outliers detected: {iso_outliers.sum().sum()}")
    
    print("  d) Local Outlier Factor Detection...")
    lof_outliers = analyzer.detect_outliers_lof(n_neighbors=20, contamination=0.1)
    print(f"     Outliers detected: {lof_outliers.sum().sum()}")
    print()
    
    # Get outlier summary
    print("2. OUTLIER SUMMARY")
    print("-" * 80)
    summary = analyzer.get_outlier_summary()
    print(summary)
    print()
    
    # Apply different treatment strategies
    print("3. OUTLIER TREATMENT")
    print("-" * 80)
    
    print("  a) Removing outliers (IQR method)...")
    removed_data = analyzer.remove_outliers(method='iqr')
    print(f"     Rows after removal: {len(removed_data)} (removed {len(df) - len(removed_data)})")
    
    print("  b) Capping outliers (Winsorization)...")
    capped_data = analyzer.cap_outliers(multiplier=1.5)
    print(f"     Data capped at IQR bounds")
    
    print("  c) Log transformation...")
    log_data = analyzer.transform_outliers_log()
    print(f"     Log transformation applied")
    
    print("  d) Box-Cox transformation...")
    boxcox_data = analyzer.transform_outliers_boxcox()
    print(f"     Box-Cox/Yeo-Johnson transformation applied")
    print()
    
    # Evaluate model performance
    print("4. MODEL PERFORMANCE COMPARISON")
    print("-" * 80)
    
    datasets_to_evaluate = ['original', 'removed_iqr', 'capped', 'log_transform', 'boxcox_transform']
    for dataset_name in datasets_to_evaluate:
        try:
            performance = analyzer.evaluate_model_performance('target', dataset_name)
            print(f"  {dataset_name.replace('_', ' ').title()}:")
            print(f"    R² Score: {performance['r2_score']:.4f} (±{performance['r2_std']:.4f})")
        except Exception as e:
            print(f"  {dataset_name}: Could not evaluate - {str(e)}")
    print()
    
    # Generate visualizations
    print("5. GENERATING VISUALIZATIONS")
    print("-" * 80)
    
    print("  Creating box plots...")
    analyzer.visualize_outliers_boxplot()
    plt.savefig('task2_boxplots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_boxplots.png")
    
    print("  Creating violin plots...")
    analyzer.visualize_outliers_violin()
    plt.savefig('task2_violin_plots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_violin_plots.png")
    
    print("  Creating scatter plots with outlier detection...")
    analyzer.visualize_outliers_scatter('feature1', 'feature2')
    plt.savefig('task2_scatter_outliers.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_scatter_outliers.png")
    
    print("  Creating 3D outlier visualization...")
    analyzer.visualize_3d_outliers('feature1', 'feature2', 'feature3', method='isolation_forest')
    plt.savefig('task2_3d_outliers.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_3d_outliers.png")
    
    print("  Creating treatment comparison plots...")
    analyzer.visualize_treatment_comparison('feature1')
    plt.savefig('task2_treatment_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_treatment_comparison.png")
    
    print("  Creating Q-Q plots...")
    analyzer.visualize_qq_plots()
    plt.savefig('task2_qq_plots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_qq_plots.png")
    print()
    
    print("=" * 80)
    print("Task 2 completed successfully!")
    print("=" * 80)
    
    summary.to_csv("task2_outlier_summary.csv")
    pd.DataFrame({name: analyzer.evaluate_model_performance("target", name)
                  for name in datasets_to_evaluate}).T.to_csv("task2_model_comparison.csv")
    return analyzer


if __name__ == "__main__":
    analyzer = main()
    plt.show()
