# Enhanced Bias & Fairness Analysis Guide

## Overview

The Nordic Privacy AI platform now includes a comprehensive, adaptive bias and fairness analysis system that works accurately across **all types of datasets**, including:

- Small datasets (< 100 samples)
- Imbalanced groups
- Multiple protected attributes
- Binary and multi-class targets
- High-cardinality features
- Missing data

## Key Enhancements

### 1. **Adaptive Fairness Thresholds**

The system automatically adjusts fairness thresholds based on dataset characteristics:

- **Sample Size Factor**: Relaxes thresholds for small sample sizes
- **Group Imbalance Factor**: Adjusts for unequal group sizes
- **Dynamic Thresholds**: 
  - Disparate Impact: 0.7-0.8 (adapts to data)
  - Statistical Parity: 0.1-0.15 (adapts to data)
  - Equal Opportunity: 0.1-0.15 (adapts to data)

### 2. **Comprehensive Fairness Metrics**

#### Individual Metrics (6 types analyzed):

1. **Disparate Impact Ratio** (4/5ths rule)
   - Measures: min_rate / max_rate across all groups
   - Fair range: 0.8 - 1.25 (or adaptive)
   - Higher weight in overall score

2. **Statistical Parity Difference**
   - Measures: Absolute difference in positive rates
   - Fair threshold: < 0.1 (or adaptive)
   - Ensures equal selection rates

3. **Equal Opportunity** (TPR equality)
   - Measures: Difference in True Positive Rates
   - Fair threshold: < 0.1 (or adaptive)
   - Ensures equal recall across groups

4. **Equalized Odds** (TPR + FPR equality)
   - Measures: Both TPR and FPR differences
   - Fair threshold: < 0.1 (or adaptive)
   - Most comprehensive fairness criterion

5. **Predictive Parity** (Precision equality)
   - Measures: Difference in precision across groups
   - Fair threshold: < 0.1
   - Ensures positive predictions are equally accurate

6. **Calibration** (FNR equality)
   - Measures: Difference in False Negative Rates
   - Fair threshold: < 0.1
   - Ensures balanced error rates

#### Group-Level Metrics (per demographic group):

- Positive Rate
- Selection Rate
- True Positive Rate (TPR/Recall/Sensitivity)
- False Positive Rate (FPR)
- True Negative Rate (TNR/Specificity)
- False Negative Rate (FNR)
- Precision (PPV)
- F1 Score
- Accuracy
- Sample Size & Distribution

### 3. **Weighted Bias Scoring**

The overall bias score (0-1, higher = more bias) is calculated using:

```python
Overall Score = Weighted Average of:
  - Disparate Impact (weight: 1.5x sample_weight)
  - Statistical Parity (weight: 1.0x sample_weight)
  - Equal Opportunity (weight: 1.0x sample_weight)
  - Equalized Odds (weight: 0.8x sample_weight)
  - Predictive Parity (weight: 0.7x sample_weight)
  - Calibration (weight: 0.7x sample_weight)
```

Sample weight = min(1.0, total_samples / 100)

### 4. **Intelligent Violation Detection**

Violations are categorized by severity:

- **CRITICAL**: di_value < 0.5, or deviation > 50%
- **HIGH**: di_value < 0.6, or deviation > 30%
- **MEDIUM**: di_value < 0.7, or deviation > 15%
- **LOW**: Minor deviations

Each violation includes:
- Affected groups
- Specific measurements
- Actionable recommendations
- Context-aware severity assessment

### 5. **Robust Data Handling**

#### Missing Values:
- Numerical: Filled with median
- Categorical: Filled with mode or 'Unknown'
- Comprehensive logging

#### Data Type Detection:
- Binary detection (0/1, Yes/No)
- Small discrete values (< 10 unique)
- High cardinality warnings (> 50 categories)
- Mixed type handling

#### Target Encoding:
- Automatic categorical → numeric conversion
- Binary value normalization
- Clear encoding maps printed

#### Class Imbalance:
- Stratified splitting when appropriate
- Minimum class size validation
- Balanced metrics calculation

### 6. **Enhanced Reporting**

Each analysis includes:

```json
{
  "overall_bias_score": 0.954,
  "fairness_metrics": {
    "Gender": {
      "disparate_impact": {
        "value": 0.276,
        "threshold": 0.8,
        "fair": false,
        "min_group": "Female",
        "max_group": "Male",
        "min_rate": 0.25,
        "max_rate": 0.906
      },
      "statistical_parity_difference": {...},
      "equal_opportunity_difference": {...},
      "equalized_odds": {...},
      "predictive_parity": {...},
      "calibration": {...},
      "attribute_fairness_score": 0.89,
      "group_metrics": {
        "Male": {
          "positive_rate": 0.906,
          "tpr": 0.95,
          "fpr": 0.03,
          "precision": 0.92,
          "f1_score": 0.93,
          "sample_size": 450
        },
        "Female": {...}
      },
      "sample_statistics": {
        "total_samples": 500,
        "min_group_size": 50,
        "max_group_size": 450,
        "imbalance_ratio": 0.11,
        "num_groups": 2
      }
    }
  },
  "fairness_violations": [
    {
      "attribute": "Gender",
      "metric": "Disparate Impact",
      "severity": "CRITICAL",
      "value": 0.276,
      "affected_groups": ["Female", "Male"],
      "message": "...",
      "recommendation": "CRITICAL: Group 'Female' has less than half the approval rate..."
    }
  ]
}
```

## Usage Examples

### Basic Analysis

```python
from ai_governance import AIGovernanceAnalyzer

# Initialize
analyzer = AIGovernanceAnalyzer()

# Analyze with protected attributes
report = analyzer.analyze(
    df=your_dataframe,
    target_column='ApprovalStatus',
    protected_attributes=['Gender', 'Age', 'Race']
)

# Check bias score
print(f"Bias Score: {report['bias_analysis']['overall_bias_score']:.1%}")

# Review violations
for violation in report['bias_analysis']['fairness_violations']:
    print(f"{violation['severity']}: {violation['message']}")
```

### With Presidio (Enhanced PII Detection)

```python
# Enable Presidio for automatic demographic detection
analyzer = AIGovernanceAnalyzer(use_presidio=True)
```

### API Usage

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@dataset.csv" \
  -F "target_column=Outcome" \
  -F "protected_attributes=Gender,Age"
```

## Interpreting Results

### Overall Bias Score

- **< 0.3**: Low bias - Excellent fairness ✅
- **0.3 - 0.5**: Moderate bias - Monitor recommended ⚠️
- **> 0.5**: High bias - Action required ❌

### Disparate Impact

- **0.8 - 1.25**: Fair (4/5ths rule satisfied)
- **< 0.8**: Disadvantaged group exists
- **> 1.25**: Advantaged group exists

### Statistical Parity

- **< 0.1**: Fair (similar positive rates)
- **> 0.1**: Groups receive different treatment

### Recommendations by Severity

#### CRITICAL
- **DO NOT DEPLOY** without remediation
- Investigate systemic bias sources
- Review training data representation
- Implement fairness constraints
- Consider re-collection if necessary

#### HIGH
- Address before deployment
- Use fairness-aware training methods
- Implement threshold optimization
- Regular monitoring required

#### MEDIUM
- Monitor closely
- Consider mitigation strategies
- Regular fairness audits
- Document findings

#### LOW
- Continue monitoring
- Maintain fairness standards
- Periodic reviews

## Best Practices

### 1. Data Collection
- Ensure representative sampling
- Balance protected groups when possible
- Document data sources
- Check for historical bias

### 2. Feature Engineering
- Avoid proxy features for protected attributes
- Check feature correlations with demographics
- Use feature importance analysis
- Consider fairness-aware feature selection

### 3. Model Training
- Use fairness-aware algorithms
- Implement fairness constraints
- Try multiple fairness definitions
- Cross-validate with fairness metrics

### 4. Post-Processing
- Threshold optimization per group
- Calibration techniques
- Reject option classification
- Regular bias audits

### 5. Monitoring
- Track fairness metrics over time
- Monitor for fairness drift
- Regular re-evaluation
- Document all findings

## Technical Details

### Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
presidio-analyzer>=2.2.0  # Optional
spacy>=3.0.0  # Optional for Presidio
```

### Performance

- Handles datasets from 50 to 1M+ rows
- Adaptive algorithms scale with data size
- Memory-efficient group comparisons
- Parallel metric calculations

### Limitations

- Requires at least 2 groups per protected attribute
- Minimum 10 samples per group recommended
- Binary classification focus (multi-class supported)
- Assumes independent test set

## Troubleshooting

### "Insufficient valid groups"
- Check protected attribute has at least 2 non-null groups
- Ensure groups appear in test set
- Increase test_size parameter

### "High cardinality warning"
- Feature has > 50 unique values
- Consider grouping categories
- May need feature engineering

### "Sample size too small"
- System adapts automatically
- Results may be less reliable
- Consider collecting more data

### "Presidio initialization failed"
- Install: `pip install presidio-analyzer spacy`
- Download model: `python -m spacy download en_core_web_sm`
- Or use `use_presidio=False`

## References

- [Fairness Definitions Explained](https://fairware.cs.umass.edu/papers/Verma.pdf)
- [4/5ths Rule (EEOC)](https://www.eeoc.gov/laws/guidance/questions-and-answers-clarify-and-provide-common-interpretation-uniform-guidelines)
- [Equalized Odds](https://arxiv.org/abs/1610.02413)
- [Fairness Through Awareness](https://arxiv.org/abs/1104.3913)

## Support

For issues or questions:
- Check logs for detailed diagnostic messages
- Review sample statistics in output
- Consult violation recommendations
- Contact: support@nordicprivacyai.com
