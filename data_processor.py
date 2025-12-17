"""
Data Processor Module
Handles quantitative analysis of phishing campaign data using Pandas
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhishingDataProcessor:
    """
    Processes phishing campaign CSV data and generates quantitative insights
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize the data processor and load data
        
        Args:
            csv_path: Path to the CSV file containing phishing campaign data
        """
        self.csv_path = csv_path
        self.df = None
        self.stats = {}
        self.load_data()
        
    def load_data(self) -> pd.DataFrame:
        """Load and validate CSV data"""
        try:
            self.df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(self.df)} records from {self.csv_path}")
            
            # Validate required columns
            required_cols = ['User_ID', 'Department', 'Template', 'Action', 'Response_Time_Sec']
            missing_cols = set(required_cols) - set(self.df.columns)
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def calculate_click_rates(self) -> Dict[str, float]:
        """
        Calculate click rates by department
        
        Returns:
            Dictionary mapping department to click rate
        """
        click_rates = {}
        
        for dept in self.df['Department'].unique():
            dept_data = self.df[self.df['Department'] == dept]
            total = len(dept_data)
            clicked = len(dept_data[dept_data['Action'] == 'Clicked'])
            click_rates[dept] = round(clicked / total if total > 0 else 0, 4)
        
        self.stats['click_rates'] = click_rates
        logger.info(f"Calculated click rates for {len(click_rates)} departments")
        
        return click_rates
    
    def calculate_template_effectiveness(self) -> pd.DataFrame:
        """
        Analyze which templates are most effective
        
        Returns:
            DataFrame with template statistics
        """
        template_stats = self.df.groupby('Template').agg({
            'User_ID': 'count',
            'Action': lambda x: (x == 'Clicked').sum(),
            'Response_Time_Sec': 'mean'
        }).rename(columns={
            'User_ID': 'total_sent',
            'Action': 'clicked',
            'Response_Time_Sec': 'avg_response_time'
        })
        
        template_stats['click_rate'] = (
            template_stats['clicked'] / template_stats['total_sent']
        ).round(4)
        
        template_stats = template_stats.sort_values('click_rate', ascending=False)
        
        self.stats['template_effectiveness'] = template_stats
        logger.info(f"Analyzed {len(template_stats)} templates")
        
        return template_stats
    
    def identify_high_risk_users(self, top_n: int = 10) -> pd.DataFrame:
        """
        Identify users who are most at risk based on their behavior
        
        Args:
            top_n: Number of top risky users to return
            
        Returns:
            DataFrame with high-risk user information
        """
        # Calculate risk score for each user
        user_stats = self.df.groupby('User_ID').agg({
            'Action': lambda x: (x == 'Clicked').sum(),
            'Response_Time_Sec': lambda x: x[x > 0].mean() if (x > 0).any() else 0,
            'Department': 'first',
            'Template': 'count'
        }).rename(columns={
            'Action': 'clicks',
            'Response_Time_Sec': 'avg_response_time',
            'Template': 'emails_received'
        })
        
        # Calculate risk score (more clicks + faster response = higher risk)
        user_stats['risk_score'] = (
            user_stats['clicks'] * 10 + 
            np.where(user_stats['avg_response_time'] > 0,
                    100 / user_stats['avg_response_time'],
                    0)
        ).round(2)
        
        high_risk = user_stats.sort_values('risk_score', ascending=False).head(top_n)
        
        self.stats['high_risk_users'] = high_risk
        logger.info(f"Identified {top_n} high-risk users")
        
        return high_risk
    
    def analyze_response_times(self) -> Dict[str, any]:
        """
        Analyze response time patterns
        
        Returns:
            Dictionary with response time statistics
        """
        clicked_data = self.df[self.df['Action'] == 'Clicked']
        
        response_stats = {
            'overall_avg': round(clicked_data['Response_Time_Sec'].mean(), 2),
            'overall_median': round(clicked_data['Response_Time_Sec'].median(), 2),
            'fastest': int(clicked_data['Response_Time_Sec'].min()),
            'slowest': int(clicked_data['Response_Time_Sec'].max()),
            'by_department': {}
        }
        
        for dept in clicked_data['Department'].unique():
            dept_times = clicked_data[clicked_data['Department'] == dept]['Response_Time_Sec']
            response_stats['by_department'][dept] = {
                'avg': round(dept_times.mean(), 2),
                'median': round(dept_times.median(), 2)
            }
        
        self.stats['response_times'] = response_stats
        logger.info("Analyzed response time patterns")
        
        return response_stats
    
    def get_department_summary(self, department: str) -> Dict[str, any]:
        """
        Get comprehensive summary for a specific department
        
        Args:
            department: Name of the department
            
        Returns:
            Dictionary with department statistics
        """
        dept_data = self.df[self.df['Department'] == department]
        
        if len(dept_data) == 0:
            return {"error": f"No data found for department: {department}"}
        
        total = len(dept_data)
        clicked = len(dept_data[dept_data['Action'] == 'Clicked'])
        ignored = len(dept_data[dept_data['Action'] == 'Ignored'])
        reported = len(dept_data[dept_data['Action'] == 'Reported'])
        
        clicked_data = dept_data[dept_data['Action'] == 'Clicked']
        avg_response = (
            round(clicked_data['Response_Time_Sec'].mean(), 2) 
            if len(clicked_data) > 0 else 0
        )
        
        # Most clicked templates
        top_templates = (
            dept_data[dept_data['Action'] == 'Clicked']
            .groupby('Template')
            .size()
            .sort_values(ascending=False)
            .head(3)
            .to_dict()
        )
        
        summary = {
            'department': department,
            'total_emails': total,
            'clicked': clicked,
            'ignored': ignored,
            'reported': reported,
            'click_rate': round(clicked / total if total > 0 else 0, 4),
            'report_rate': round(reported / total if total > 0 else 0, 4),
            'avg_response_time': avg_response,
            'top_clicked_templates': top_templates,
            'unique_users': dept_data['User_ID'].nunique()
        }
        
        return summary
    
    def generate_full_report(self) -> Dict[str, any]:
        """
        Generate comprehensive analysis report
        
        Returns:
            Dictionary containing all analysis results
        """
        report = {
            'overview': {
                'total_emails': len(self.df),
                'total_users': self.df['User_ID'].nunique(),
                'total_departments': self.df['Department'].nunique(),
                'total_templates': self.df['Template'].nunique(),
                'overall_click_rate': round(
                    len(self.df[self.df['Action'] == 'Clicked']) / len(self.df), 4
                ),
                'overall_report_rate': round(
                    len(self.df[self.df['Action'] == 'Reported']) / len(self.df), 4
                )
            },
            'click_rates': self.calculate_click_rates(),
            'template_effectiveness': self.calculate_template_effectiveness().to_dict('index'),
            'high_risk_users': self.identify_high_risk_users().to_dict('index'),
            'response_times': self.analyze_response_times()
        }
        
        logger.info("Generated full analysis report")
        return report


if __name__ == "__main__":
    # Test the data processor
    processor = PhishingDataProcessor("sample_phishing_data.csv")
    
    print("\n=== Click Rates by Department ===")
    click_rates = processor.calculate_click_rates()
    for dept, rate in click_rates.items():
        print(f"{dept}: {rate:.2%}")
    
    print("\n=== Template Effectiveness ===")
    templates = processor.calculate_template_effectiveness()
    print(templates.head())
    
    print("\n=== High Risk Users ===")
    high_risk = processor.identify_high_risk_users(5)
    print(high_risk)
    
    print("\n=== Finance Department Summary ===")
    finance_summary = processor.get_department_summary("Finance")
    print(finance_summary)
