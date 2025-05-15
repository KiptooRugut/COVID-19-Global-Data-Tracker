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
        
def _clean_data(self):
        """Data cleaning with modern pandas methods"""
        # Forward fill and then fill remaining NAs with 0
        for col in self.metrics:
            if col in self.df.columns:
                self.df[col] = self.df[col].ffill().fillna(0)
        
        # Calculate derived metrics
        self.df['case_fatality_rate'] = (self.df['total_deaths'] / self.df['total_cases'] * 100).replace([np.inf, -np.inf], 0)
        
        # Calculate rolling metrics
        for country in self.countries:
            country_mask = self.df['location'] == country
            for metric in ['new_cases', 'new_deaths']:
                if metric in self.df.columns:
                    self.df.loc[country_mask, f'{metric}_7day_avg'] = (
                        self.df.loc[country_mask, metric]
                        .rolling(window=7).mean()
                    )


def analyze(self):
        """Perform analysis with available data"""
        if not hasattr(self, 'df') or self.df.empty:
            print("No data to analyze!")
            return None

        latest_date = self.df['date'].max()
        latest_data = self.df[self.df['date'] == latest_date]
        
        self.analysis = {
            "latest_date": latest_date,
            "peak_cases": self._find_peaks('new_cases'),
            "peak_deaths": self._find_peaks('new_deaths'),
            "total_stats": latest_data.groupby('location')[['total_cases', 'total_deaths']].last(),
            "fatality_rates": self._calculate_fatality_rates(),
            "per_million_stats": self._calculate_per_million_stats(),
            "trend_analysis": self._analyze_trends()
        }
        return self.analysis


def _find_peaks(self, metric):
        """Find peak values and dates for a metric"""
        if metric not in self.df.columns:
            return None
            
        peaks = {}
        for country in self.countries:
            country_data = self.df[self.df['location'] == country]
            if not country_data.empty:
                max_idx = country_data[metric].idxmax()
                peaks[country] = {
                    "value": country_data.loc[max_idx, metric],
                    "date": country_data.loc[max_idx, 'date'],
                    "per_million": country_data.loc[max_idx, f'{metric}_per_million'] 
                        if f'{metric}_per_million' in country_data.columns else None
                }
        return peaks

def _calculate_fatality_rates(self):
        """Calculate case fatality rates"""
        if 'total_deaths' not in self.df.columns or 'total_cases' not in self.df.columns:
            return None
            
        latest = self.df.groupby('location').last()
        return (latest['total_deaths'] / latest['total_cases'] * 100).round(2)


def _calculate_per_million_stats(self):
        """Calculate per million statistics"""
        stats = {}
        for metric in ['cases', 'deaths']:
            total_col = f'total_{metric}'
            per_mil_col = f'total_{metric}_per_million'
            
            if per_mil_col in self.df.columns:
                latest = self.df.groupby('location').last()
                stats[metric] = latest[[total_col, per_mil_col]]
        return stats if stats else None


def _analyze_trends(self):
        """Analyze recent trends (last 30 days)"""
        recent = self.df[self.df['date'] > (self.df['date'].max() - pd.Timedelta(days=30))]
        
        trend_metrics = [m for m in self.metrics if m.startswith(('new_', 'total_'))]
        return recent.groupby('location')[trend_metrics].agg(['mean', 'max', 'min', 'last'])


def visualize(self):
        """Generate visualizations with available data"""
        if not hasattr(self, 'df') or self.df.empty:
            print("No data to visualize!")
            return

        print("Creating visualizations...")
        with PdfPages(self.output_pdf) as pdf:
            self._plot_timeseries(pdf)
            self._plot_peaks(pdf)
            
            if self.analysis['fatality_rates'] is not None:
                self._plot_fatality_rates(pdf)
                
            if self.analysis['per_million_stats'] is not None:
                self._plot_per_million(pdf)


def _plot_timeseries(self, pdf):
        """Plot time series of key metrics"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Cases
        if 'new_cases_7day_avg' in self.df.columns:
            sns.lineplot(
                data=self.df, x='date', y='new_cases_7day_avg', 
                hue='location', ax=axes[0]
            )
            axes[0].set_title('7-Day Average of New Cases')
            axes[0].set_ylabel('Cases')
        
        # Deaths
        if 'new_deaths_7day_avg' in self.df.columns:
            sns.lineplot(
                data=self.df, x='date', y='new_deaths_7day_avg', 
                hue='location', ax=axes[1]
            )
            axes[1].set_title('7-Day Average of New Deaths')
            axes[1].set_ylabel('Deaths')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.savefig(f"{self.output_png_prefix}Timeseries.png")
        plt.close()


def _plot_peaks(self, pdf):
        """Plot peak cases and deaths"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Peak cases
        if self.analysis['peak_cases']:
            peak_cases = pd.DataFrame.from_dict(
                self.analysis['peak_cases'], 
                orient='index'
            )['value']
            peak_cases.plot(kind='bar', ax=axes[0], color='orange')
            axes[0].set_title('Peak Daily Cases')
        
        # Peak deaths
        if self.analysis['peak_deaths']:
            peak_deaths = pd.DataFrame.from_dict(
                self.analysis['peak_deaths'], 
                orient='index'
            )['value']
            peak_deaths.plot(kind='bar', ax=axes[1], color='red')
            axes[1].set_title('Peak Daily Deaths')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.savefig(f"{self.output_png_prefix}Peaks.png")
        plt.close()


def _plot_fatality_rates(self, pdf):
        """Plot case fatality rates"""
        fig, ax = plt.subplots(figsize=(10, 6))
        self.analysis['fatality_rates'].plot(
            kind='bar', ax=ax, color='purple'
        )
        ax.set_title('Case Fatality Rate (%)')
        ax.set_ylabel('Fatality Rate %')
        plt.xticks(rotation=45)
        
        pdf.savefig(fig)
        plt.savefig(f"{self.output_png_prefix}FatalityRates.png")
        plt.close()