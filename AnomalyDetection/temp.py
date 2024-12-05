import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import synthetic_data_generator

class AccountsPayableAnomalyDetection:
    def __init__(self):
        """
        Initialize the Accounts Payable Anomaly Detection Pipeline
        """
        self.raw_data = None
        self.processed_data = None
        self.scaled_data = None
        self.anomalies = None

    def generate_synthetic_dataset(self, n_samples=1000):
        """
        Generate synthetic accounts payable dataset
        
        Features might include:
        - Invoice amount
        - Payment delay
        - Vendor frequency
        - Payment amount variation
        - Discount utilization
        """
        np.random.seed(42)
        
        # Normal data generation
        normal_invoices = {
            'invoice_amount': np.random.normal(5000, 1500, n_samples),
            'payment_delay': np.random.normal(30, 10, n_samples),
            'vendor_frequency': np.random.normal(5, 2, n_samples),
            'payment_amount_variation': np.random.normal(0.05, 0.02, n_samples),
            'discount_utilization': np.random.normal(0.8, 0.15, n_samples)
        }
        
        # Anomaly data generation (20% of samples)
        anomaly_invoices = {
            'invoice_amount': np.random.normal(20000, 5000, int(n_samples * 0.2)),
            'payment_delay': np.random.normal(90, 30, int(n_samples * 0.2)),
            'vendor_frequency': np.random.normal(1, 0.5, int(n_samples * 0.2)),
            'payment_amount_variation': np.random.normal(0.3, 0.1, int(n_samples * 0.2)),
            'discount_utilization': np.random.normal(0.1, 0.05, int(n_samples * 0.2))
        }
        
        # Combine datasets
        normal_df = pd.DataFrame(normal_invoices)
        anomaly_df = pd.DataFrame(anomaly_invoices)
        
        # Add label column
        normal_df['is_anomaly'] = 0
        anomaly_df['is_anomaly'] = 1
        
        self.raw_data = pd.concat([normal_df, anomaly_df], ignore_index=True)
        return self.raw_data

    def preprocess_data(self):
        """
        Preprocess the data:
        - Handle missing values
        - Scale features
        - Perform dimensionality reduction
        """
        # Remove any missing values
        self.processed_data = self.raw_data.dropna()
        
        # Separate features for scaling
        features = ['invoice_amount', 'payment_delay', 'vendor_frequency', 
                    'payment_amount_variation', 'discount_utilization']
        
        # Scale the features
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(self.processed_data[features])
        
        return self.scaled_data

    def apply_anomaly_detection_methods(self):
        """
        Apply multiple anomaly detection algorithms:
        1. DBSCAN
        2. One-Class SVM
        3. Isolation Forest
        """
        results = {}
        
        # DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        results['DBSCAN'] = dbscan.fit_predict(self.scaled_data)
        
        # One-Class SVM
        ocsvm = OneClassSVM(kernel='rbf', nu=0.1)
        results['OneClassSVM'] = ocsvm.fit_predict(self.scaled_data)
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        results['IsolationForest'] = iso_forest.fit_predict(self.scaled_data)
        
        return results

    def visualize_anomalies(self, anomaly_results):
        """
        Visualize anomalies using dimensionality reduction and plotting
        """
        # PCA for dimensionality reduction
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(self.scaled_data)
        
        plt.figure(figsize=(15, 5))
        
        # Plot for each method
        for i, (method, labels) in enumerate(anomaly_results.items(), 1):
            plt.subplot(1, 3, i)
            plt.title(f'Anomalies - {method}')
            
            # Plot normal and anomalous points
            normal_mask = labels != -1
            anomaly_mask = labels == -1
            
            plt.scatter(reduced_data[normal_mask, 0], reduced_data[normal_mask, 1], 
                        c='blue', label='Normal', alpha=0.7)
            plt.scatter(reduced_data[anomaly_mask, 0], reduced_data[anomaly_mask, 1], 
                        c='red', label='Anomaly', alpha=0.7)
            
            plt.legend()
        
        plt.tight_layout()
        plt.show()

    def generate_anomaly_report(self, anomaly_results):
        """
        Generate a comprehensive anomaly detection report
        """
        report = {}
        for method, labels in anomaly_results.items():
            anomaly_count = np.sum(labels == -1)
            total_samples = len(labels)
            anomaly_percentage = (anomaly_count / total_samples) * 100
            
            report[method] = {
                'Total Samples': total_samples,
                'Anomalies Detected': anomaly_count,
                'Anomaly Percentage': f'{anomaly_percentage:.2f}%'
            }
        
        return report

    def run_pipeline(self):
        """
        Run the complete anomaly detection pipeline
        """
        # Generate synthetic dataset
        self.generate_synthetic_dataset()
        
        # Preprocess data
        self.preprocess_data()
        
        # Apply anomaly detection methods
        anomaly_results = self.apply_anomaly_detection_methods()
        
        # Visualize anomalies
        self.visualize_anomalies(anomaly_results)
        
        # Generate report
        anomaly_report = self.generate_anomaly_report(anomaly_results)
        
        return anomaly_report

# Example usage
ap_anomaly_detector = AccountsPayableAnomalyDetection()
anomaly_report = ap_anomaly_detector.run_pipeline()
print(anomaly_report)