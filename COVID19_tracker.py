import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from datetime import datetime
import os
import warnings

class COVID19Tracker:
    def __init__(self):
        # Countries to analyze (reduced to match your sample data)
        self.countries = ['Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 
            'Argentina', 'Armenia', 'Australia', 'Austria','Bahamas', 'Bahrain', 
            'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 
            'Bhutan', 'Bolivia','Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia',
            'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy','Jamaica', 'Japan', 'Jordan',
            'Kenya','Kiribati', 'Korea, South', 'Kuwait', 'Kyrgyzstan',
            ]  # From your sample data
        
        # Metrics that exist in your sample data
        self.metrics = [
            'new_cases',
            'new_cases_smoothed',
            'new_deaths',
            'new_deaths_smoothed',
            'total_cases',
            'total_deaths',
            'total_cases_per_million',
            'new_cases_per_million',
            'new_cases_smoothed_per_million',
            'total_deaths_per_million',
            'new_deaths_per_million',
            'new_deaths_smoothed_per_million'
        ]
        
        self.output_pdf = "COVID19_Analysis_Report.pdf"
        self.output_png_prefix = "COVID19_Visualization_"


def load_data(self, filepath):
        """Load data from CSV file with enhanced error handling"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Data file not found at: {filepath}")
                
            print(f"Loading data from {filepath}...")
            
            # Suppress warnings during data loading
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.df = pd.read_csv(filepath, parse_dates=['date'])
            
            # Verify required columns exist
            required_cols = ['date', 'location'] + self.metrics
            missing_cols = [col for col in required_cols 
                          if col not in self.df.columns]
            if missing_cols:
                print(f"Note: Missing some columns in data: {missing_cols}")
                # Remove missing metrics from our analysis
                self.metrics = [col for col in self.metrics if col in self.df.columns]
            
            # Filter to selected countries
            self.df = self.df[self.df['location'].isin(self.countries)]
            if self.df.empty:
                raise ValueError("No data found for selected countries")
                
            self._clean_data()
            print(f"Data loaded successfully with {len(self.df)} records")
            print(f"Available metrics: {self.metrics}")
            return True
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False