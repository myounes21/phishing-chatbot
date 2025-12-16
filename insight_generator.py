"""
Insight Generator Module
Converts quantitative analysis into qualitative, human-readable insights
"""

import json
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InsightGenerator:
    """
    Generates natural language insights from phishing campaign data analysis
    """
    
    def __init__(self, data_processor):
        """
        Initialize the insight generator
        
        Args:
            data_processor: Instance of PhishingDataProcessor
        """
        self.data_processor = data_processor
        self.insights = []
        
    def generate_department_insights(self) -> List[Dict[str, Any]]:
        """
        Generate insights about department vulnerabilities
        
        Returns:
            List of insight dictionaries
        """
        click_rates = self.data_processor.calculate_click_rates()
        insights = []
        
        # Sort departments by click rate
        sorted_depts = sorted(click_rates.items(), key=lambda x: x[1], reverse=True)
        
        for dept, rate in sorted_depts:
            dept_summary = self.data_processor.get_department_summary(dept)
            
            # Determine severity level
            if rate >= 0.30:
                severity = "HIGH"
                description = "critically vulnerable"
            elif rate >= 0.20:
                severity = "MEDIUM"
                description = "moderately vulnerable"
            else:
                severity = "LOW"
                description = "relatively secure"
            
            # Get top template
            top_templates = dept_summary.get('top_clicked_templates', {})
            top_template = list(top_templates.keys())[0] if top_templates else "various templates"
            
            # Generate insight text
            insight_text = (
                f"The {dept} department shows a {rate:.1%} click rate, making it "
                f"{description} to phishing attacks. Employees are particularly susceptible "
                f"to '{top_template}' style phishing emails, with {dept_summary['clicked']} "
                f"out of {dept_summary['total_emails']} emails clicked. "
            )
            
            if dept_summary['avg_response_time'] > 0:
                insight_text += (
                    f"The average response time of {dept_summary['avg_response_time']:.0f} seconds "
                    f"suggests {'quick, impulsive clicking behavior' if dept_summary['avg_response_time'] < 60 else 'thoughtful but ultimately unsuccessful verification attempts'}. "
                )
            
            if dept_summary['report_rate'] > 0:
                insight_text += (
                    f"Positively, {dept_summary['report_rate']:.1%} of emails were properly reported, "
                    f"indicating some security awareness within the team."
                )
            else:
                insight_text += (
                    f"Concerningly, no phishing emails were reported, suggesting lack of security awareness training."
                )
            
            insight = {
                'insight_id': f"DEPT_{dept.upper()}_001",
                'category': 'department_vulnerability',
                'source_type': 'phishing_insight',
                'department': dept,
                'severity': severity,
                'click_rate': rate,
                'text': insight_text,
                'metadata': dept_summary
            }
            
            insights.append(insight)
        
        self.insights.extend(insights)
        logger.info(f"Generated {len(insights)} department insights")
        return insights
    
    def generate_template_insights(self) -> List[Dict[str, Any]]:
        """
        Generate insights about template effectiveness
        
        Returns:
            List of insight dictionaries
        """
        template_stats = self.data_processor.calculate_template_effectiveness()
        insights = []
        
        for template, stats in template_stats.iterrows():
            click_rate = stats['click_rate']
            total_sent = stats['total_sent']
            clicked = stats['clicked']
            avg_time = stats['avg_response_time']
            
            # Determine effectiveness
            if click_rate >= 0.40:
                effectiveness = "HIGHLY EFFECTIVE"
                description = "extremely successful"
            elif click_rate >= 0.25:
                effectiveness = "EFFECTIVE"
                description = "moderately successful"
            else:
                effectiveness = "LOW EFFECTIVENESS"
                description = "less successful"
            
            # Analyze psychological triggers
            triggers = self._identify_psychological_triggers(template)
            trigger_text = f"This template exploits {triggers}. " if triggers else ""
            
            # Generate insight text
            insight_text = (
                f"The '{template}' template is {description} with a {click_rate:.1%} click rate "
                f"({clicked} clicks out of {total_sent} sent). {trigger_text}"
            )
            
            if avg_time > 0:
                if avg_time < 30:
                    insight_text += (
                        f"The rapid average response time of {avg_time:.0f} seconds indicates "
                        f"strong urgency triggers that bypass critical thinking. "
                    )
                elif avg_time < 60:
                    insight_text += (
                        f"The average response time of {avg_time:.0f} seconds suggests "
                        f"effective social engineering that creates a sense of time pressure. "
                    )
                else:
                    insight_text += (
                        f"Despite a longer consideration time ({avg_time:.0f} seconds), "
                        f"users still clicked, indicating sophisticated social engineering. "
                    )
            
            insight = {
                'insight_id': f"TMPL_{template.replace(' ', '_').upper()}_001",
                'category': 'template_effectiveness',
                'source_type': 'phishing_insight',
                'template': template,
                'effectiveness': effectiveness,
                'click_rate': click_rate,
                'text': insight_text,
                'metadata': {
                    'total_sent': int(total_sent),
                    'clicked': int(clicked),
                    'avg_response_time': float(avg_time),
                    'psychological_triggers': triggers
                }
            }
            
            insights.append(insight)
        
        self.insights.extend(insights)
        logger.info(f"Generated {len(insights)} template insights")
        return insights
    
    def generate_user_risk_insights(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Generate insights about high-risk users
        
        Args:
            top_n: Number of top risky users to analyze
            
        Returns:
            List of insight dictionaries
        """
        high_risk_users = self.data_processor.identify_high_risk_users(top_n)
        insights = []
        
        for user_id, stats in high_risk_users.iterrows():
            clicks = stats['clicks']
            dept = stats['Department']
            avg_time = stats['avg_response_time']
            risk_score = stats['risk_score']
            
            # Determine risk level
            if clicks >= 3:
                risk_level = "CRITICAL"
                description = "extremely high-risk"
            elif clicks >= 2:
                risk_level = "HIGH"
                description = "high-risk"
            else:
                risk_level = "MODERATE"
                description = "moderate-risk"
            
            # Generate insight text
            insight_text = (
                f"User {user_id} from {dept} department is {description} with {clicks} "
                f"phishing email clicks (risk score: {risk_score:.2f}). "
            )
            
            if avg_time > 0:
                if avg_time < 30:
                    insight_text += (
                        f"Their extremely fast average response time ({avg_time:.0f}s) indicates "
                        f"they click without verification, making them prime targets for attackers. "
                    )
                else:
                    insight_text += (
                        f"Despite taking {avg_time:.0f} seconds to respond, they still clicked, "
                        f"suggesting difficulty distinguishing legitimate from malicious emails. "
                    )
            
            insight_text += (
                f"Immediate security awareness training is recommended, with focus on "
                f"email verification techniques and recognizing social engineering tactics."
            )
            
            insight = {
                'insight_id': f"USER_{user_id}_RISK_001",
                'category': 'user_risk_profile',
                'source_type': 'phishing_insight',
                'user_id': user_id,
                'risk_level': risk_level,
                'risk_score': risk_score,
                'text': insight_text,
                'metadata': {
                    'clicks': int(clicks),
                    'department': dept,
                    'avg_response_time': float(avg_time),
                    'emails_received': int(stats['emails_received'])
                }
            }
            
            insights.append(insight)
        
        self.insights.extend(insights)
        logger.info(f"Generated {len(insights)} user risk insights")
        return insights
    
    def generate_behavioral_insights(self) -> List[Dict[str, Any]]:
        """
        Generate insights about behavioral patterns across the organization
        
        Returns:
            List of insight dictionaries
        """
        response_stats = self.data_processor.analyze_response_times()
        click_rates = self.data_processor.calculate_click_rates()
        insights = []
        
        # Overall behavioral pattern
        overall_avg = response_stats['overall_avg']
        overall_median = response_stats['overall_median']
        
        insight_text = (
            f"Organization-wide analysis reveals an average response time of {overall_avg:.0f} seconds "
            f"(median: {overall_median:.0f}s) for phishing emails that were clicked. "
        )
        
        if overall_avg < 45:
            insight_text += (
                f"This rapid response pattern indicates widespread vulnerability to time-pressure tactics. "
                f"The organization would benefit from training emphasizing the 'pause and verify' approach."
            )
        else:
            insight_text += (
                f"While employees take some time to consider, they still fall victim, indicating "
                f"sophisticated phishing attempts that appear legitimate even under scrutiny."
            )
        
        # Compare fastest vs slowest responders
        fastest = response_stats['fastest']
        slowest = response_stats['slowest']
        
        insight_text += (
            f" Response times range from {fastest}s to {slowest}s, showing diverse "
            f"behavioral patterns across the organization."
        )
        
        behavioral_insight = {
            'insight_id': 'BEHAVIORAL_ORG_001',
            'category': 'behavioral_insights',
            'source_type': 'phishing_insight',
            'text': insight_text,
            'metadata': response_stats
        }
        
        insights.append(behavioral_insight)
        
        # Cross-department comparison
        dept_response_times = response_stats['by_department']
        sorted_depts = sorted(
            dept_response_times.items(),
            key=lambda x: x[1]['avg']
        )
        
        fastest_dept = sorted_depts[0][0] if sorted_depts else None
        slowest_dept = sorted_depts[-1][0] if sorted_depts else None
        
        if fastest_dept and slowest_dept:
            comparison_text = (
                f"Behavioral comparison across departments shows significant variation. "
                f"{fastest_dept} has the fastest average response time "
                f"({dept_response_times[fastest_dept]['avg']:.0f}s), suggesting impulsive behavior, "
                f"while {slowest_dept} takes longer to respond "
                f"({dept_response_times[slowest_dept]['avg']:.0f}s), indicating more deliberation "
                f"but ultimately unsuccessful verification."
            )
            
            comparison_insight = {
                'insight_id': 'BEHAVIORAL_COMP_001',
                'category': 'behavioral_insights',
                'source_type': 'phishing_insight',
                'text': comparison_text,
                'metadata': {
                    'fastest_dept': fastest_dept,
                    'slowest_dept': slowest_dept,
                    'dept_response_times': dept_response_times
                }
            }
            
            insights.append(comparison_insight)
        
        self.insights.extend(insights)
        logger.info(f"Generated {len(insights)} behavioral insights")
        return insights
    
    def _identify_psychological_triggers(self, template_name: str) -> str:
        """
        Identify psychological triggers in template names
        
        Args:
            template_name: Name of the phishing template
            
        Returns:
            Description of psychological triggers
        """
        triggers = []
        
        template_lower = template_name.lower()
        
        if 'urgent' in template_lower:
            triggers.append('urgency and time pressure')
        if 'ceo' in template_lower or 'executive' in template_lower:
            triggers.append('authority and hierarchy')
        if 'password' in template_lower or 'reset' in template_lower:
            triggers.append('security concerns and fear')
        if 'invoice' in template_lower or 'payment' in template_lower:
            triggers.append('financial responsibility')
        if 'package' in template_lower or 'delivery' in template_lower:
            triggers.append('anticipation and curiosity')
        if 'bank' in template_lower or 'account' in template_lower:
            triggers.append('financial security fears')
        if 'payroll' in template_lower or 'benefits' in template_lower:
            triggers.append('personal financial interest')
        
        if triggers:
            return ', '.join(triggers)
        else:
            return 'general social engineering tactics'
    
    def generate_all_insights(self) -> List[Dict[str, Any]]:
        """
        Generate all types of insights
        
        Returns:
            List of all generated insights
        """
        self.insights = []
        
        self.generate_department_insights()
        self.generate_template_insights()
        self.generate_user_risk_insights()
        self.generate_behavioral_insights()
        
        logger.info(f"Generated total of {len(self.insights)} insights")
        return self.insights
    
    def save_insights(self, filepath: str):
        """
        Save insights to JSON file
        
        Args:
            filepath: Path to save the insights
        """
        with open(filepath, 'w') as f:
            json.dump(self.insights, f, indent=2)
        
        logger.info(f"Saved {len(self.insights)} insights to {filepath}")
    
    def get_insights_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get insights filtered by category
        
        Args:
            category: Category to filter by
            
        Returns:
            List of insights matching the category
        """
        return [i for i in self.insights if i['category'] == category]


if __name__ == "__main__":
    from data_processor import PhishingDataProcessor
    
    # Test the insight generator
    processor = PhishingDataProcessor("sample_phishing_data.csv")
    processor.load_data()
    
    generator = InsightGenerator(processor)
    insights = generator.generate_all_insights()
    
    print(f"\n=== Generated {len(insights)} Total Insights ===\n")
    
    for insight in insights[:5]:  # Show first 5
        print(f"ID: {insight['insight_id']}")
        print(f"Category: {insight['category']}")
        print(f"Text: {insight['text']}")
        print("-" * 80)
    
    # Save insights
    generator.save_insights("phishing_insights.json")
