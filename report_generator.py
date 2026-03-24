# report_generator.py - Report Generation (Fixed)

import pandas as pd
import base64
from datetime import datetime

class ReportGenerator:
    def __init__(self):
        pass
    
    def generate_html_report(self, results_df, dataset_info, feature_importance, best_model):
        """Generate HTML report"""
        
        # Convert results_df to HTML safely
        if results_df is not None and len(results_df) > 0:
            # Remove any problematic columns
            display_cols = []
            for col in results_df.columns:
                if col not in ['Model Object', 'Accuracy_Score']:
                    display_cols.append(col)
            
            results_html = results_df[display_cols].to_html(index=False, escape=False)
        else:
            results_html = "<p>No results available</p>"
        
        # Convert feature importance to HTML
        if feature_importance is not None and len(feature_importance) > 0:
            # Handle different column names
            if 'Feature' in feature_importance.columns:
                imp_display = feature_importance[['Feature']].copy()
                if 'Importance' in feature_importance.columns:
                    imp_display['Importance'] = feature_importance['Importance']
                elif 'MI_Score' in feature_importance.columns:
                    imp_display['Importance'] = feature_importance['MI_Score']
                else:
                    imp_display['Importance'] = 0
                
                feature_html = imp_display.head(10).to_html(index=False)
            else:
                feature_html = "<p>Feature importance not available</p>"
        else:
            feature_html = "<p>Feature importance not available</p>"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CardioAI Report</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 40px;
                }}
                
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 20px;
                    padding: 40px;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                }}
                
                h1 {{
                    color: #667eea;
                    text-align: center;
                    font-size: 2.5rem;
                    margin-bottom: 10px;
                }}
                
                h2 {{
                    color: #764ba2;
                    border-bottom: 3px solid #667eea;
                    padding-bottom: 10px;
                    margin-top: 30px;
                    margin-bottom: 20px;
                }}
                
                .date {{
                    text-align: center;
                    color: #666;
                    margin-bottom: 30px;
                    padding-bottom: 20px;
                    border-bottom: 1px solid #eee;
                }}
                
                .info-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                
                .info-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 15px;
                    text-align: center;
                }}
                
                .info-card h3 {{
                    font-size: 2rem;
                    margin-bottom: 5px;
                }}
                
                .info-card p {{
                    font-size: 0.9rem;
                    opacity: 0.9;
                }}
                
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    background: white;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                
                th {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 12px;
                    text-align: left;
                    font-weight: 600;
                }}
                
                td {{
                    padding: 10px 12px;
                    border-bottom: 1px solid #eee;
                }}
                
                tr:hover {{
                    background: #f5f5f5;
                }}
                
                .best-model {{
                    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 15px;
                    text-align: center;
                    margin: 20px 0;
                }}
                
                .best-model h3 {{
                    font-size: 1.5rem;
                    margin-bottom: 10px;
                }}
                
                .recommendations {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 15px;
                    margin: 20px 0;
                }}
                
                .recommendations ul {{
                    list-style: none;
                    padding-left: 0;
                }}
                
                .recommendations li {{
                    padding: 8px 0;
                    padding-left: 25px;
                    position: relative;
                }}
                
                .recommendations li:before {{
                    content: "✓";
                    color: #10b981;
                    position: absolute;
                    left: 0;
                    font-weight: bold;
                }}
                
                .footer {{
                    text-align: center;
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                    color: #666;
                    font-size: 0.85rem;
                }}
                
                @media (max-width: 768px) {{
                    body {{
                        padding: 20px;
                    }}
                    .container {{
                        padding: 20px;
                    }}
                    h1 {{
                        font-size: 1.8rem;
                    }}
                    .info-grid {{
                        grid-template-columns: 1fr;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>❤️ CardioAI - Heart Disease Prediction Report</h1>
                <div class="date">
                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
                
                <h2>📊 Dataset Information</h2>
                <div class="info-grid">
                    <div class="info-card">
                        <h3>{dataset_info.get('samples', 0)}</h3>
                        <p>Total Samples</p>
                    </div>
                    <div class="info-card">
                        <h3>{dataset_info.get('features', 0)}</h3>
                        <p>Total Features</p>
                    </div>
                    <div class="info-card">
                        <h3>{dataset_info.get('positive', 0)}</h3>
                        <p>Positive Cases</p>
                    </div>
                    <div class="info-card">
                        <h3>{dataset_info.get('negative', 0)}</h3>
                        <p>Negative Cases</p>
                    </div>
                </div>
                
                <h2>🤖 Model Performance Comparison</h2>
                {results_html}
                
                <div class="best-model">
                    <h3>🏆 Best Performing Model</h3>
                    <p style="font-size: 1.2rem; margin-top: 10px;">{best_model}</p>
                </div>
                
                <h2>📈 Feature Importance (Top 10)</h2>
                {feature_html}
                
                <h2>💊 Medical Recommendations</h2>
                <div class="recommendations">
                    <ul>
                        <li><strong>Regular Exercise:</strong> At least 30 minutes of moderate activity daily</li>
                        <li><strong>Healthy Diet:</strong> Low saturated fats, high fiber, plenty of fruits and vegetables</li>
                        <li><strong>Regular Check-ups:</strong> Annual health screening and cardiac check-ups</li>
                        <li><strong>Stress Management:</strong> Practice meditation, yoga, or deep breathing exercises</li>
                        <li><strong>Adequate Sleep:</strong> 7-8 hours of quality sleep per night</li>
                        <li><strong>No Smoking:</strong> Avoid tobacco and limit alcohol consumption</li>
                        <li><strong>Monitor BP:</strong> Regular blood pressure monitoring</li>
                        <li><strong>Medication Adherence:</strong> Take prescribed medicines regularly</li>
                    </ul>
                </div>
                
                <div class="footer">
                    <p>⚠️ This report is generated by CardioAI Pro - An AI-powered heart disease prediction system.</p>
                    <p>📌 For medical advice, please consult a qualified healthcare professional.</p>
                    <p>🔬 Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        """
        return html
    
    def generate_csv_report(self, results_df, dataset_info, feature_importance):
        """Generate CSV report"""
        import io
        
        output = io.StringIO()
        
        # Write header
        output.write("CardioAI Heart Disease Prediction Report\n")
        output.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Dataset info
        output.write("DATASET INFORMATION\n")
        output.write(f"Total Samples,{dataset_info.get('samples', 0)}\n")
        output.write(f"Total Features,{dataset_info.get('features', 0)}\n")
        output.write(f"Target Column,{dataset_info.get('target', 'N/A')}\n")
        output.write(f"Positive Cases,{dataset_info.get('positive', 0)}\n")
        output.write(f"Negative Cases,{dataset_info.get('negative', 0)}\n\n")
        
        # Model results
        if results_df is not None and len(results_df) > 0:
            output.write("MODEL PERFORMANCE\n")
            results_df.to_csv(output, index=False)
            output.write("\n")
        
        # Feature importance
        if feature_importance is not None and len(feature_importance) > 0:
            output.write("FEATURE IMPORTANCE\n")
            feature_importance.to_csv(output, index=False)
        
        return output.getvalue()