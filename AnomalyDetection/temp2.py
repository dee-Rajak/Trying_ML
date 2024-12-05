import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class MultiFeatureAnomalyDetector:
    def __init__(self, dataframe):
        """
        Initialize the anomaly detector with the input dataframe
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            Input dataset to detect anomalies
        """
        self.df = dataframe.copy()
        self.anomaly_results = {}
    
    def detect_numerical_anomalies(self, column, z_threshold=3):
        """
        Detect anomalies in numerical columns using Z-score method
        
        Parameters:
        -----------
        column : str
            Name of the numerical column to check
        z_threshold : float, optional
            Z-score threshold for anomaly detection (default 3)
        
        Returns:
        --------
        pandas.Series
            Boolean series indicating anomalies
        """
        # Convert to numeric, removing any '-' signs
        numeric_data = self.df[column].apply(lambda x: float(str(x).rstrip('-')))
        
        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(numeric_data))
        
        return z_scores > z_threshold
    
    def detect_categorical_anomalies(self, column):
        """
        Detect anomalies in categorical columns using frequency-based approach
        
        Parameters:
        -----------
        column : str
            Name of the categorical column to check
        
        Returns:
        --------
        pandas.Series
            Boolean series indicating anomalies
        """
        # Calculate value counts
        value_counts = self.df[column].value_counts()
        
        # Identify rare categories (less than 1% of total)
        rare_categories = value_counts[value_counts / len(self.df) < 0.01].index
        
        return self.df[column].isin(rare_categories)
    
    def detect_date_anomalies(self, column):
        """
        Detect anomalies in date columns
        
        Parameters:
        -----------
        column : str
            Name of the date column to check
        
        Returns:
        --------
        pandas.Series
            Boolean series indicating anomalies
        """
        # Convert to datetime
        dates = pd.to_datetime(self.df[column])
        
        # Calculate quartiles
        Q1 = dates.quantile(0.25)
        Q3 = dates.quantile(0.75)
        IQR = Q3 - Q1
        
        # Define anomaly bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return (dates < lower_bound) | (dates > upper_bound)
    
    def run_anomaly_detection(self):
        """
        Run comprehensive anomaly detection across all features
        
        Returns:
        --------
        dict
            Dictionary of anomaly detection results for each feature
        """
        # Numerical columns
        numerical_cols = ['Gross Amount', 'VAT Amount']
        for col in numerical_cols:
            self.anomaly_results[col] = self.detect_numerical_anomalies(col)
        
        # Categorical columns
        categorical_cols = ['Supplier Name', 'Currency Code', 'Debit/Credit']
        for col in categorical_cols:
            self.anomaly_results[col] = self.detect_categorical_anomalies(col)
        
        # Date columns
        date_cols = ['Invoice Date', 'Posting Date', 'Due Date']
        for col in date_cols:
            self.anomaly_results[col] = self.detect_date_anomalies(col)
        
        return self.anomaly_results
    
    def generate_detailed_anomaly_report(self):
        """
        Generate a detailed report of anomalies for each record
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with detailed anomaly information
        """
        # Run anomaly detection if not already done
        if not self.anomaly_results:
            self.run_anomaly_detection()
        
        # Create a list to store anomaly details
        detailed_anomalies = []
        
        # Iterate through each record
        for idx in range(len(self.df)):
            record_anomalies = []
            
            # Check each feature for anomalies
            for feature, anomalies in self.anomaly_results.items():
                if anomalies[idx]:
                    record_anomalies.append(feature)
            
            # If the record has any anomalies
            if record_anomalies:
                detailed_anomalies.append({
                    'Internal Reference': self.df.loc[idx, 'Internal Reference'],
                    'Invoice Number': self.df.loc[idx, 'Invoice Number'],
                    'Anomalous Features': record_anomalies
                })
        
        # Convert to DataFrame
        anomaly_df = pd.DataFrame(detailed_anomalies)
        return anomaly_df
    
    def visualize_anomalies(self):
        """
        Visualize the distribution of anomalies across different features
        """
        # Count anomalies per feature
        anomaly_counts = {feature: anomalies.sum() for feature, anomalies in self.anomaly_results.items()}
        
        # Plotting
        plt.figure(figsize=(12, 6))
        plt.bar(anomaly_counts.keys(), anomaly_counts.values())
        plt.title('Number of Anomalies per Feature')
        plt.xlabel('Features')
        plt.ylabel('Number of Anomalies')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

# Usage example
def main():
    # Load the synthetic dataset
    df = pd.read_csv('accounts_payable_synthetic_dataset.csv')
    
    # Initialize anomaly detector
    detector = MultiFeatureAnomalyDetector(df)
    
    # Run anomaly detection
    anomaly_results = detector.run_anomaly_detection()
    
    # Visualize anomalies
    detector.visualize_anomalies()
    
    # Generate detailed anomaly report
    detailed_anomalies = detector.generate_detailed_anomaly_report()
    
    # Display detailed anomalies
    print("\nDetailed Anomalies:")
    print(detailed_anomalies)
    
    # Optional: Save detailed anomalies to CSV
    detailed_anomalies.to_csv('detailed_anomalies.csv', index=False)

if __name__ == "__main__":
    main()