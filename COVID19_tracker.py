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