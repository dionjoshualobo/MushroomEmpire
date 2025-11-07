# Nordic Privacy AI 🛡️# Nordic Privacy AI 🛡️# AI Governance Module



**AI-Powered GDPR Compliance & Privacy Protection Platform**



A comprehensive solution for AI governance, bias detection, risk assessment, and automated PII cleaning with GDPR compliance. Built for Nordic ecosystems and beyond.**AI-Powered GDPR Compliance & Privacy Protection Platform**A Python package for detecting bias and analyzing risks in machine learning models. Provides comprehensive fairness metrics, privacy risk assessment, and ethical AI evaluation.



[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)

[![Next.js](https://img.shields.io/badge/Next.js-14.2+-black.svg)](https://nextjs.org/)A comprehensive solution for AI governance, bias detection, risk assessment, and automated PII cleaning with GDPR compliance. Built for Nordic ecosystems and beyond.## Features

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)



---

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)### 🎯 Bias Detection

## 🚀 Quick Start

[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)- **Fairness Metrics**: Disparate Impact, Statistical Parity Difference, Equal Opportunity Difference

### Prerequisites

- Python 3.8+[![Next.js](https://img.shields.io/badge/Next.js-14.2+-black.svg)](https://nextjs.org/)- **Demographic Analysis**: Group-wise performance evaluation

- Node.js 18+

- GPU (optional, for faster processing)[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)- **Violation Detection**: Automatic flagging with severity levels



### Installation



1. **Clone the repository**---### 🛡️ Risk Assessment

```powershell

git clone https://github.com/PlatypusPus/MushroomEmpire.git- **Privacy Risks**: PII detection, GDPR compliance, data exposure analysis

cd MushroomEmpire

```## 🚀 Quick Start- **Ethical Risks**: Fairness, transparency, accountability, social impact



2. **Install Python dependencies**- **Compliance Risks**: Regulatory adherence (GDPR, CCPA, AI Act)

```powershell

pip install -r requirements.txt### Prerequisites- **Data Quality**: Missing data, class imbalance, outlier detection

python -m spacy download en_core_web_sm

```- Python 3.8+



3. **Install frontend dependencies**- Node.js 18+### 🤖 Machine Learning

```powershell

cd frontend- GPU (optional, for faster processing)- Generalized classification model (works with any dataset)

npm install

cd ..- Auto-detection of feature types and protected attributes

```

### Installation- Comprehensive performance metrics

### Running the Application

- Feature importance analysis

1. **Start the FastAPI backend** (Terminal 1)

```powershell1. **Clone the repository**

python start_api.py

``````powershell## Installation

Backend runs at: **http://localhost:8000**

git clone https://github.com/PlatypusPus/MushroomEmpire.git

2. **Start the Next.js frontend** (Terminal 2)

```powershellcd MushroomEmpire```bash

cd frontend

npm run dev```pip install -r requirements.txt

```

Frontend runs at: **http://localhost:3000**```



3. **Access the application**2. **Install Python dependencies**

   - Frontend UI: http://localhost:3000

   - Try It Page: http://localhost:3000/try```powershellOr install as a package:

   - API Documentation: http://localhost:8000/docs

   - Health Check: http://localhost:8000/healthpip install -r requirements.txt



---python -m spacy download en_core_web_sm```bash



## 📋 Features```pip install -e .



### 🎯 AI Governance & Bias Detection```

- **Fairness Metrics**: Disparate Impact, Statistical Parity, Equal Opportunity

- **Demographic Analysis**: Group-wise performance evaluation3. **Install frontend dependencies**

- **Violation Detection**: Automatic flagging with severity levels (HIGH/MEDIUM/LOW)

- **Model Performance**: Comprehensive ML metrics (accuracy, precision, recall, F1)```powershell## Quick Start



### 🛡️ Privacy Risk Assessmentcd frontend/nordic-privacy-ai

- **Privacy Risks**: PII detection, GDPR compliance scoring, data exposure analysis

- **Ethical Risks**: Fairness, transparency, accountability evaluationnpm install```python

- **Compliance Risks**: Regulatory adherence (GDPR, CCPA, AI Act)

- **Data Quality**: Missing data, class imbalance, outlier detectioncd ../..from ai_governance import AIGovernanceAnalyzer



### 🧹 Automated Data Cleaning```

- **PII Detection**: Email, phone, SSN, credit cards, IP addresses, and more

- **GPU Acceleration**: CUDA-enabled for 10x faster processing# Initialize analyzer

- **GDPR Compliance**: Automatic anonymization with audit trails

- **Smart Anonymization**: Context-aware masking and pseudonymization### Running the Applicationanalyzer = AIGovernanceAnalyzer()



### 🌐 Modern Web Interface

- **Drag & Drop Upload**: Intuitive CSV file handling

- **Real-time Processing**: Live feedback and progress tracking1. **Start the FastAPI backend** (Terminal 1)# Run complete analysis

- **Interactive Dashboards**: Visualize bias metrics, risk scores, and results

- **Report Downloads**: JSON reports, cleaned CSV, and audit logs```powershellreport = analyzer.analyze(



---python start_api.py    data_path='your_data.csv',



## 🏗️ Project Structure```    target_column='target',



```Backend runs at: **http://localhost:8000**    protected_attributes=['gender', 'age', 'race']

MushroomEmpire/

├── api/                          # FastAPI Backend)

│   ├── main.py                   # Application entry point

│   ├── routers/2. **Start the Next.js frontend** (Terminal 2)

│   │   ├── analyze.py           # POST /api/analyze - AI Governance

│   │   └── clean.py             # POST /api/clean - Data Cleaning```powershell# Access results

│   └── utils/                    # Helper utilities

│cd frontend/nordic-privacy-aiprint(f"Bias Score: {report['summary']['overall_bias_score']:.3f}")

├── ai_governance/                # Core AI Governance Module

│   ├── __init__.py              # AIGovernanceAnalyzer classnpm run devprint(f"Risk Level: {report['summary']['risk_level']}")

│   ├── data_processor.py        # Data preprocessing

│   ├── model_trainer.py         # ML model training```print(f"Model Accuracy: {report['summary']['model_accuracy']:.3f}")

│   ├── bias_analyzer.py         # Bias detection engine

│   ├── risk_analyzer.py         # Risk assessment engineFrontend runs at: **http://localhost:3000**

│   └── report_generator.py      # JSON report generation

│# Save report

├── data_cleaning/                # Data Cleaning Module

│   ├── __init__.py              # DataCleaner class3. **Access the application**analyzer.save_report(report, 'governance_report.json')

│   ├── cleaner.py               # PII detection & anonymization

│   └── config.py                # PII patterns & GDPR rules   - Frontend UI: http://localhost:3000```

│

├── frontend/                     # Next.js Frontend   - API Documentation: http://localhost:8000/docs

│   ├── app/                     # App Router pages

│   │   ├── page.tsx            # Landing page   - Health Check: http://localhost:8000/health## Module Structure

│   │   └── try/page.tsx        # Try it page (workflow UI)

│   ├── components/

│   │   └── try/

│   │       ├── CenterPanel.tsx  # File upload & results---```

│   │       ├── Sidebar.tsx      # Workflow tabs

│   │       └── ChatbotPanel.tsx # AI assistantai_governance/

│   └── lib/

│       ├── api.ts              # TypeScript API client## 📋 Features├── __init__.py              # Main API

│       └── indexeddb.ts        # Browser caching utilities

│├── data_processor.py        # Data preprocessing

├── Datasets/                     # Sample datasets

│   └── loan_data.csv            # Example: Loan approval dataset### 🎯 AI Governance & Bias Detection├── model_trainer.py         # ML model training

│

├── reports/                      # Generated reports (auto-created)- **Fairness Metrics**: Disparate Impact, Statistical Parity, Equal Opportunity├── bias_analyzer.py         # Bias detection

│   ├── governance_report_*.json

│   ├── cleaned_*.csv- **Demographic Analysis**: Group-wise performance evaluation├── risk_analyzer.py         # Risk assessment

│   └── cleaning_audit_*.json

│- **Violation Detection**: Automatic flagging with severity levels (HIGH/MEDIUM/LOW)└── report_generator.py      # Report generation

├── start_api.py                 # Backend startup script

├── setup.py                     # Package configuration- **Model Performance**: Comprehensive ML metrics (accuracy, precision, recall, F1)```

├── requirements.txt             # Python dependencies

└── README.md                    # This file

```

### 🛡️ Privacy Risk Assessment## API Reference

---

- **Privacy Risks**: PII detection, GDPR compliance scoring, data exposure analysis

## 📡 API Reference

- **Ethical Risks**: Fairness, transparency, accountability evaluation### AIGovernanceAnalyzer

### Base URL

```- **Compliance Risks**: Regulatory adherence (GDPR, CCPA, AI Act)

http://localhost:8000

```- **Data Quality**: Missing data, class imbalance, outlier detectionMain class for running AI governance analysis.



### Endpoints



#### **POST /api/analyze**### 🧹 Automated Data Cleaning```python

Analyze dataset for bias, fairness, and risk assessment.

- **PII Detection**: Email, phone, SSN, credit cards, IP addresses, and moreanalyzer = AIGovernanceAnalyzer()

**Request:**

```bash- **GPU Acceleration**: CUDA-enabled for 10x faster processing

curl -X POST "http://localhost:8000/api/analyze" \

  -F "file=@Datasets/loan_data.csv"- **GDPR Compliance**: Automatic anonymization with audit trails# Analyze from DataFrame

```

- **Smart Anonymization**: Context-aware masking and pseudonymizationreport = analyzer.analyze_dataframe(

**Response:**

```json    df=dataframe,

{

  "status": "success",### 🌐 Modern Web Interface    target_column='target',

  "filename": "loan_data.csv",

  "dataset_info": {- **Drag & Drop Upload**: Intuitive CSV file handling    protected_attributes=['gender', 'age']

    "rows": 1000,

    "columns": 15- **Real-time Processing**: Live feedback and progress tracking)

  },

  "model_performance": {- **Interactive Dashboards**: Visualize bias metrics, risk scores, and results

    "accuracy": 0.85,

    "precision": 0.82,- **Report Downloads**: JSON reports, cleaned CSV, and audit logs# Analyze from file

    "recall": 0.88,

    "f1_score": 0.85report = analyzer.analyze(

  },

  "bias_metrics": {---    data_path='data.csv',

    "overall_bias_score": 0.23,

    "violations_detected": []    target_column='target',

  },

  "risk_assessment": {## 🏗️ Project Structure    protected_attributes=['gender', 'age']

    "overall_risk_score": 0.35,

    "privacy_risks": [],)

    "ethical_risks": []

  },``````

  "recommendations": [

    "[HIGH] Privacy: Remove PII columns before deployment",MushroomEmpire/

    "[MEDIUM] Fairness: Monitor demographic parity over time"

  ],├── api/                          # FastAPI Backend### Individual Components

  "report_file": "/reports/governance_report_20251107_123456.json"

}│   ├── main.py                   # Application entry point

```

│   ├── routers/```python

#### **POST /api/clean**

Detect and anonymize PII in datasets.│   │   ├── analyze.py           # POST /api/analyze - AI Governancefrom ai_governance import (



**Request:**│   │   └── clean.py             # POST /api/clean - Data Cleaning    DataProcessor,

```bash

curl -X POST "http://localhost:8000/api/clean" \│   └── utils/                    # Helper utilities    GeneralizedModelTrainer,

  -F "file=@Datasets/loan_data.csv"

```│    BiasAnalyzer,



**Response:**├── ai_governance/                # Core AI Governance Module    RiskAnalyzer,

```json

{│   ├── __init__.py              # AIGovernanceAnalyzer class    ReportGenerator

  "status": "success",

  "dataset_info": {│   ├── data_processor.py        # Data preprocessing)

    "original_rows": 1000,

    "original_columns": 15,│   ├── model_trainer.py         # ML model training

    "cleaned_rows": 1000,

    "cleaned_columns": 13│   ├── bias_analyzer.py         # Bias detection engine# Process data

  },

  "summary": {│   ├── risk_analyzer.py         # Risk assessment engineprocessor = DataProcessor(df)

    "columns_removed": ["ssn", "email"],

    "columns_anonymized": ["phone", "address"],│   └── report_generator.py      # JSON report generationprocessor.target_column = 'target'

    "total_cells_affected": 2847

  },│processor.protected_attributes = ['gender', 'age']

  "pii_detections": {

    "EMAIL": 1000,├── cleaning.py                   # Core PII detection & anonymizationprocessor.prepare_data()

    "PHONE": 987,

    "SSN": 1000├── cleaning_config.py           # Configuration for data cleaning

  },

  "gdpr_compliance": [├── test_cleaning.py             # Unit tests for cleaning module# Train model

    "Article 5(1)(c) - Data minimization",

    "Article 17 - Right to erasure",│trainer = GeneralizedModelTrainer(

    "Article 25 - Data protection by design"

  ],├── frontend/nordic-privacy-ai/  # Next.js Frontend    processor.X_train,

  "files": {

    "cleaned_csv": "/reports/cleaned_20251107_123456.csv",│   ├── app/                     # App Router pages    processor.X_test,

    "audit_report": "/reports/cleaning_audit_20251107_123456.json"

  }│   │   ├── page.tsx            # Landing page    processor.y_train,

}

```│   │   └── try/page.tsx        # Try it page (workflow UI)    processor.y_test,



#### **GET /health**│   ├── components/    processor.feature_names

Health check endpoint with GPU status.

│   │   └── try/)

**Response:**

```json│   │       ├── CenterPanel.tsx  # File upload & resultstrainer.train()

{

  "status": "healthy",│   │       ├── Sidebar.tsx      # Workflow tabstrainer.evaluate()

  "version": "1.0.0",

  "gpu_available": true│   │       └── ChatbotPanel.tsx # AI assistant

}

```│   └── lib/# Analyze bias



#### **GET /reports/{filename}**│       ├── api.ts              # TypeScript API clientbias_analyzer = BiasAnalyzer(

Download generated reports and cleaned files.

│       └── indexeddb.ts        # Browser caching utilities    processor.X_test,

---

│    processor.y_test,

## 🔧 Configuration

├── Datasets/                     # Sample datasets    trainer.y_pred,

### Environment Variables

│   └── loan_data.csv            # Example: Loan approval dataset    processor.df,

Create `.env` file in `frontend/`:

```env│    processor.protected_attributes,

NEXT_PUBLIC_API_URL=http://localhost:8000

```├── reports/                      # Generated reports (auto-created)    processor.target_column



### CORS Configuration│   ├── governance_report_*.json)



Edit `api/main.py` to add production domains:│   ├── cleaned_*.csvbias_results = bias_analyzer.analyze()

```python

origins = [│   └── cleaning_audit_*.json

    "http://localhost:3000",

    "https://your-production-domain.com"│# Assess risks

]

```├── start_api.py                 # Backend startup scriptrisk_analyzer = RiskAnalyzer(



### GPU Acceleration├── setup.py                     # Package configuration    processor.df,



GPU is automatically detected and used if available. To force CPU mode:├── requirements.txt             # Python dependencies    trainer.results,

```python

# In data_cleaning/cleaner.py or api endpoints└── README.md                    # This file    bias_results,

DataCleaner(use_gpu=False)

``````    processor.protected_attributes,



---    processor.target_column



## 🧪 Testing---)



### Test the Backendrisk_results = risk_analyzer.analyze()

```powershell

# Test analyze endpoint## 📡 API Reference

curl -X POST "http://localhost:8000/api/analyze" -F "file=@Datasets/loan_data.csv"

# Generate report

# Test clean endpoint

curl -X POST "http://localhost:8000/api/clean" -F "file=@Datasets/loan_data.csv"### Base URLreport_gen = ReportGenerator(



# Check health```    trainer.results,

curl http://localhost:8000/health

```http://localhost:8000    bias_results,



### Run Unit Tests```    risk_results,

```powershell

# Test cleaning module    processor.df

python test_cleaning.py

### Endpoints)

# Run all tests (if pytest configured)

pytestreport = report_gen.generate_report()

```

#### **POST /api/analyze**```

---

Analyze dataset for bias, fairness, and risk assessment.

## 📊 Usage Examples

## Report Structure

### Python SDK Usage

**Request:**

```python

from ai_governance import AIGovernanceAnalyzer```bashThe module generates comprehensive JSON reports:



# Initialize analyzercurl -X POST "http://localhost:8000/api/analyze" \

analyzer = AIGovernanceAnalyzer()

  -F "file=@Datasets/loan_data.csv"```json

# Analyze dataset

report = analyzer.analyze(```{

    data_path='Datasets/loan_data.csv',

    target_column='loan_approved',  "metadata": {

    protected_attributes=['gender', 'age', 'race']

)**Response:**    "report_id": "unique_id",



# Print results```json    "generated_at": "timestamp",

print(f"Bias Score: {report['summary']['overall_bias_score']:.3f}")

print(f"Risk Level: {report['summary']['risk_level']}"){    "dataset_info": {}

print(f"Model Accuracy: {report['summary']['model_accuracy']:.3f}")

  "status": "success",  },

# Save report

analyzer.save_report(report, 'my_report.json')  "filename": "loan_data.csv",  "summary": {

```

  "dataset_info": {    "overall_bias_score": 0.0-1.0,

### Data Cleaning Usage

    "rows": 1000,    "overall_risk_score": 0.0-1.0,

```python

from data_cleaning import DataCleaner    "columns": 15    "risk_level": "LOW|MEDIUM|HIGH",



# Initialize cleaner with GPU  },    "model_accuracy": 0.0-1.0,

cleaner = DataCleaner(use_gpu=True)

  "model_performance": {    "fairness_violations_count": 0

# Load and clean data

df = cleaner.load_data('Datasets/loan_data.csv')    "accuracy": 0.85,  },

cleaned_df, audit = cleaner.anonymize_pii(df)

    "precision": 0.82,  "model_performance": {},

# Save results

cleaner.save_cleaned_data(cleaned_df, 'cleaned_output.csv')    "recall": 0.88,  "bias_analysis": {},

cleaner.save_audit_report(audit, 'audit_report.json')

```    "f1_score": 0.85  "risk_assessment": {},



### Frontend Integration  },  "key_findings": [],



```typescript  "bias_metrics": {  "recommendations": []

import { analyzeDataset, cleanDataset } from '@/lib/api';

    "overall_bias_score": 0.23,}

// Analyze uploaded file

const handleAnalyze = async (file: File) => {    "violations_detected": []```

  const result = await analyzeDataset(file);

  console.log('Bias Score:', result.bias_metrics.overall_bias_score);  },

  console.log('Download:', result.report_file);

};  "risk_assessment": {## Metrics Interpretation



// Clean uploaded file    "overall_risk_score": 0.35,

const handleClean = async (file: File) => {

  const result = await cleanDataset(file);    "privacy_risks": [],### Bias Score (0-1, lower is better)

  console.log('Cells anonymized:', result.summary.total_cells_affected);

  console.log('Download cleaned:', result.files.cleaned_csv);    "ethical_risks": []- **0.0 - 0.3**: Low bias ✅

};

```  },- **0.3 - 0.5**: Moderate bias ⚠️



---  "recommendations": [- **0.5 - 1.0**: High bias ❌



## 📈 Metrics Interpretation    "[HIGH] Privacy: Remove PII columns before deployment",



### Bias Score (0-1, lower is better)    "[MEDIUM] Fairness: Monitor demographic parity over time"### Risk Score (0-1, lower is better)

- **0.0 - 0.3**: ✅ Low bias - Good fairness

- **0.3 - 0.5**: ⚠️ Moderate bias - Monitoring recommended  ],- **0.0 - 0.4**: LOW risk ✅

- **0.5 - 1.0**: ❌ High bias - Immediate action required

  "report_file": "/reports/governance_report_20251107_123456.json"- **0.4 - 0.7**: MEDIUM risk ⚠️

### Risk Score (0-1, lower is better)

- **0.0 - 0.4**: ✅ LOW risk}- **0.7 - 1.0**: HIGH risk ❌

- **0.4 - 0.7**: ⚠️ MEDIUM risk

- **0.7 - 1.0**: ❌ HIGH risk```



### Fairness Metrics### Fairness Metrics

- **Disparate Impact**: Fair range 0.8 - 1.25

- **Statistical Parity**: Fair threshold < 0.1#### **POST /api/clean**- **Disparate Impact**: Fair range 0.8 - 1.25

- **Equal Opportunity**: Fair threshold < 0.1

Detect and anonymize PII in datasets.- **Statistical Parity**: Fair threshold < 0.1

---

- **Equal Opportunity**: Fair threshold < 0.1

## 🛠️ Technology Stack

**Request:**

### Backend

- **FastAPI** - Modern Python web framework```bash## Requirements

- **scikit-learn** - Machine learning

- **spaCy** - NLP for PII detectioncurl -X POST "http://localhost:8000/api/clean" \

- **PyTorch** - GPU acceleration (optional)

- **pandas** - Data processing  -F "file=@Datasets/loan_data.csv"- Python 3.8+



### Frontend```- pandas >= 2.0.0

- **Next.js 14** - React framework with App Router

- **TypeScript** - Type safety- numpy >= 1.24.0

- **Tailwind CSS** - Styling

- **IndexedDB** - Browser storage**Response:**- scikit-learn >= 1.3.0



---```json



## 🤝 Contributing{See `requirements.txt` for complete list.



Contributions are welcome! Please follow these steps:  "status": "success",



1. Fork the repository  "dataset_info": {## Integration Examples

2. Create a feature branch (`git checkout -b feature/AmazingFeature`)

3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)    "original_rows": 1000,

4. Push to the branch (`git push origin feature/AmazingFeature`)

5. Open a Pull Request    "original_columns": 15,### FastAPI Backend



---    "cleaned_rows": 1000,



## 📝 License    "cleaned_columns": 13```python



This project is licensed under the MIT License - see the LICENSE file for details.  },from fastapi import FastAPI, UploadFile



---  "summary": {from ai_governance import AIGovernanceAnalyzer



## 🎓 Citation    "columns_removed": ["ssn", "email"],



If you use this project in your research or work, please cite:    "columns_anonymized": ["phone", "address"],app = FastAPI()



```bibtex    "total_cells_affected": 2847analyzer = AIGovernanceAnalyzer()

@software{nordic_privacy_ai,

  title = {Nordic Privacy AI - GDPR Compliance & AI Governance Platform},  },

  author = {PlatypusPus},

  year = {2025},  "pii_detections": {@app.post("/analyze")

  url = {https://github.com/PlatypusPus/MushroomEmpire}

}    "EMAIL": 1000,async def analyze(file: UploadFile, target: str, protected: list):

```

    "PHONE": 987,    df = pd.read_csv(file.file)

---

    "SSN": 1000    report = analyzer.analyze_dataframe(df, target, protected)

## 📧 Support

  },    return report

- **Issues**: [GitHub Issues](https://github.com/PlatypusPus/MushroomEmpire/issues)

- **Discussions**: [GitHub Discussions](https://github.com/PlatypusPus/MushroomEmpire/discussions)  "gdpr_compliance": [```



---    "Article 5(1)(c) - Data minimization",



## 🙏 Acknowledgments    "Article 17 - Right to erasure",### Flask Backend



- Built for Nordic ecosystems (BankID, MitID, Suomi.fi)    "Article 25 - Data protection by design"

- Inspired by GDPR, CCPA, and EU AI Act requirements

- Developed during a hackathon prototype  ],```python



---  "files": {from flask import Flask, request, jsonify



**Made with ❤️ by the Nordic Privacy AI Team**    "cleaned_csv": "/reports/cleaned_20251107_123456.csv",from ai_governance import AIGovernanceAnalyzer


    "audit_report": "/reports/cleaning_audit_20251107_123456.json"

  }app = Flask(__name__)

}analyzer = AIGovernanceAnalyzer()

```

@app.route('/analyze', methods=['POST'])

#### **GET /health**def analyze():

Health check endpoint with GPU status.    file = request.files['file']

    df = pd.read_csv(file)

**Response:**    report = analyzer.analyze_dataframe(

```json        df,

{        request.form['target'],

  "status": "healthy",        request.form.getlist('protected')

  "version": "1.0.0",    )

  "gpu_available": true    return jsonify(report)

}```

```

## License

#### **GET /reports/{filename}**

Download generated reports and cleaned files.MIT License



---## Contributing



## 🔧 ConfigurationContributions welcome! Please open an issue or submit a pull request.



### Environment Variables## Citation



Create `.env` file in `frontend/nordic-privacy-ai/`:If you use this module in your research or project, please cite:

```env

NEXT_PUBLIC_API_URL=http://localhost:8000```

```AI Governance Module - Bias Detection and Risk Analysis

https://github.com/PlatypusPus/MushroomEmpire

### CORS Configuration```


Edit `api/main.py` to add production domains:
```python
origins = [
    "http://localhost:3000",
    "https://your-production-domain.com"
]
```

### GPU Acceleration

GPU is automatically detected and used if available. To force CPU mode:
```python
# In cleaning.py or api endpoints
DataCleaner(use_gpu=False)
```

---

## 🧪 Testing

### Test the Backend
```powershell
# Test analyze endpoint
curl -X POST "http://localhost:8000/api/analyze" -F "file=@Datasets/loan_data.csv"

# Test clean endpoint
curl -X POST "http://localhost:8000/api/clean" -F "file=@Datasets/loan_data.csv"

# Check health
curl http://localhost:8000/health
```

### Run Unit Tests
```powershell
# Test cleaning module
python test_cleaning.py

# Run all tests (if pytest configured)
pytest
```

---

## 📊 Usage Examples

### Python SDK Usage

```python
from ai_governance import AIGovernanceAnalyzer

# Initialize analyzer
analyzer = AIGovernanceAnalyzer()

# Analyze dataset
report = analyzer.analyze(
    data_path='Datasets/loan_data.csv',
    target_column='loan_approved',
    protected_attributes=['gender', 'age', 'race']
)

# Print results
print(f"Bias Score: {report['summary']['overall_bias_score']:.3f}")
print(f"Risk Level: {report['summary']['risk_level']}")
print(f"Model Accuracy: {report['summary']['model_accuracy']:.3f}")

# Save report
analyzer.save_report(report, 'my_report.json')
```

### Data Cleaning Usage

```python
from cleaning import DataCleaner

# Initialize cleaner with GPU
cleaner = DataCleaner(use_gpu=True)

# Load and clean data
df = cleaner.load_data('Datasets/loan_data.csv')
cleaned_df, audit = cleaner.anonymize_pii(df)

# Save results
cleaner.save_cleaned_data(cleaned_df, 'cleaned_output.csv')
cleaner.save_audit_report(audit, 'audit_report.json')
```

### Frontend Integration

```typescript
import { analyzeDataset, cleanDataset } from '@/lib/api';

// Analyze uploaded file
const handleAnalyze = async (file: File) => {
  const result = await analyzeDataset(file);
  console.log('Bias Score:', result.bias_metrics.overall_bias_score);
  console.log('Download:', result.report_file);
};

// Clean uploaded file
const handleClean = async (file: File) => {
  const result = await cleanDataset(file);
  console.log('Cells anonymized:', result.summary.total_cells_affected);
  console.log('Download cleaned:', result.files.cleaned_csv);
};
```

---

## 📈 Metrics Interpretation

### Bias Score (0-1, lower is better)
- **0.0 - 0.3**: ✅ Low bias - Good fairness
- **0.3 - 0.5**: ⚠️ Moderate bias - Monitoring recommended
- **0.5 - 1.0**: ❌ High bias - Immediate action required

### Risk Score (0-1, lower is better)
- **0.0 - 0.4**: ✅ LOW risk
- **0.4 - 0.7**: ⚠️ MEDIUM risk
- **0.7 - 1.0**: ❌ HIGH risk

### Fairness Metrics
- **Disparate Impact**: Fair range 0.8 - 1.25
- **Statistical Parity**: Fair threshold < 0.1
- **Equal Opportunity**: Fair threshold < 0.1

---

## 🛠️ Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **scikit-learn** - Machine learning
- **spaCy** - NLP for PII detection
- **PyTorch** - GPU acceleration (optional)
- **pandas** - Data processing

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **IndexedDB** - Browser storage

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🎓 Citation

If you use this project in your research or work, please cite:

```bibtex
@software{nordic_privacy_ai,
  title = {Nordic Privacy AI - GDPR Compliance & AI Governance Platform},
  author = {PlatypusPus},
  year = {2025},
  url = {https://github.com/PlatypusPus/MushroomEmpire}
}
```

---

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/PlatypusPus/MushroomEmpire/issues)
- **Discussions**: [GitHub Discussions](https://github.com/PlatypusPus/MushroomEmpire/discussions)

---

## 🙏 Acknowledgments

- Built for Nordic ecosystems (BankID, MitID, Suomi.fi)
- Inspired by GDPR, CCPA, and EU AI Act requirements
- Developed during a hackathon prototype

---

**Made with ❤️ by the Nordic Privacy AI Team**
