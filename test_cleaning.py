"""
Test script for data cleaning module
Tests general PII + Nordic-specific PII detection with automatic report generation
"""

import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_cleaning import DataCleaner


def test_basic_cleaning():
    """Test basic cleaning functionality"""
    print("\n" + "="*70)
    print("TEST 1: Basic PII Detection on Loan Dataset")
    print("="*70)
    
    # Load loan data
    df = pd.read_csv('Datasets/loan_data.csv')
    print(f"\n✓ Loaded dataset: {len(df)} rows × {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")
    
    # Initialize cleaner
    cleaner = DataCleaner(df)
    
    # Run cleaning in non-interactive mode (auto-apply strategies)
    print("\n🔍 Running PII detection...")
    cleaned_df, audit_report = cleaner.clean(
        risky_features=None,  # Auto-detect all
        interactive=False,  # Non-interactive for testing
        scan_all_cells=True
    )
    
    # Display results
    cleaner.print_audit_summary(audit_report)
    
    return cleaned_df, audit_report


def test_with_risky_features():
    """Test cleaning with specific risky features flagged"""
    print("\n" + "="*70)
    print("TEST 2: Cleaning with Pre-Flagged Risky Features")
    print("="*70)
    
    # Load loan data
    df = pd.read_csv('Datasets/loan_data.csv')
    
    # Simulate risky features from RiskAnalyzer
    risky_features = ['person_education', 'loan_intent', 'person_home_ownership']
    
    print(f"\n⚠️  Risky features flagged by RiskAnalyzer: {risky_features}")
    
    # Initialize cleaner
    cleaner = DataCleaner(df)
    
    # Run cleaning on flagged features only
    cleaned_df, audit_report = cleaner.clean(
        risky_features=risky_features,
        interactive=False,
        scan_all_cells=False  # Only scan risky columns
    )
    
    # Display results
    cleaner.print_audit_summary(audit_report)
    
    return cleaned_df, audit_report


def test_with_synthetic_pii():
    """Test with synthetic general PII data"""
    print("\n" + "="*70)
    print("TEST 3: General PII Detection (US/International)")
    print("="*70)
    
    # Create test DataFrame with obvious PII
    test_data = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'email': [
            'john.doe@example.com',
            'alice.smith@company.org',
            'bob.jones@email.com',
            'carol.white@test.net',
            'dave.brown@sample.com'
        ],
        'phone': [
            '+1-555-123-4567',
            '555-234-5678',
            '(555) 345-6789',
            '555.456.7890',
            '5555678901'
        ],
        'ssn': [
            '123-45-6789',
            '234-56-7890',
            '345-67-8901',
            '456-78-9012',
            '567-89-0123'
        ],
        'notes': [
            'Customer called from 192.168.1.1',
            'Contact via email: test@example.com',
            'SSN verified: 111-22-3333',
            'Previous address: 123 Main St, Boston',
            'Phone backup: 555-999-8888'
        ],
        'amount': [1000, 2000, 1500, 3000, 2500]
    })
    
    print(f"\n✓ Created synthetic dataset with general PII:")
    print(test_data.head())
    
    # Initialize cleaner
    cleaner = DataCleaner(test_data)
    
    # Run cleaning
    cleaned_df, audit_report = cleaner.clean(
        risky_features=None,
        interactive=False,
        scan_all_cells=True
    )
    
    print("\n🔒 Cleaned dataset:")
    print(cleaned_df.head())
    
    # Display results
    cleaner.print_audit_summary(audit_report)
    
    # Save outputs
    os.makedirs('output', exist_ok=True)
    cleaner.save_cleaned_data(cleaned_df, 'output/general_pii_cleaned.csv')
    cleaner.save_audit_report(audit_report, 'output/general_pii_audit.json')
    
    # Generate reports
    print("\n📊 Generating explainability reports...")
    cleaner.save_simple_report(audit_report, 'output/general_pii_simple_report.json', 'General PII Test Dataset')
    cleaner.save_detailed_report(audit_report, 'output/general_pii_detailed_report.json', 'General PII Test Dataset')
    
    return cleaned_df, audit_report


def test_nordic_pii():
    """Test with Nordic-specific PII data (Finnish, Swedish, Norwegian, Danish)"""
    print("\n" + "="*70)
    print("TEST 4: Nordic PII Detection (FI, SE, NO, DK)")
    print("="*70)
    
    # Create Nordic healthcare test dataset
    nordic_data = pd.DataFrame({
        'patient_id': [1001, 1002, 1003, 1004, 1005],
        'name': ['Matti Virtanen', 'Anna Andersson', 'Lars Nilsen', 'Sofie Jensen', 'Björn Eriksson'],
        'henkilotunnus': ['010190-123A', '150685-456B', '310795+789C', '220503A234D', '111280-567E'],  # Finnish
        'personnummer': [None, '850615-4567', None, None, '801211-8901'],  # Swedish
        'fodselsnummer': [None, None, '010190 12345', None, None],  # Norwegian
        'cpr_nummer': [None, None, None, '010190-1234', None],  # Danish
        'email': ['matti.v@example.fi', 'anna.a@healthcare.se', 'lars.n@clinic.no', 'sofie.j@hospital.dk', 'bjorn.e@care.se'],
        'phone': ['+358 50 123 4567', '+46 70 234 5678', '+47 91 345 6789', '+45 20 456 7890', '+46 73 567 8901'],
        'diagnosis_code': ['J06.9', 'I10', 'E11.9', 'M54.5', 'F41.1'],
        'age': [35, 39, 29, 22, 45],
        'gender': ['M', 'F', 'M', 'F', 'M'],
        'treatment_outcome': ['Recovered', 'Ongoing', 'Recovered', 'Ongoing', 'Recovered']
    })
    
    print(f"\n✓ Created Nordic healthcare dataset:")
    print(f"  - Finnish Henkilötunnus (HETU)")
    print(f"  - Swedish Personnummer")
    print(f"  - Norwegian Fødselsnummer")
    print(f"  - Danish CPR-nummer")
    print(f"  - Nordic phone numbers (+358, +46, +47, +45)")
    print(f"  - Nordic email domains (.fi, .se, .no, .dk)")
    print()
    print(nordic_data.to_string())
    
    # Initialize cleaner (Nordic recognizers loaded automatically)
    cleaner = DataCleaner(nordic_data)
    
    # Run cleaning
    cleaned_df, audit_report = cleaner.clean(
        risky_features=None,
        interactive=False,
        scan_all_cells=True
    )
    
    print("\n🔒 Cleaned dataset (Nordic IDs anonymized):")
    print(cleaned_df.to_string())
    
    # Display results
    cleaner.print_audit_summary(audit_report)
    
    # Save outputs
    os.makedirs('output', exist_ok=True)
    cleaner.save_cleaned_data(cleaned_df, 'output/nordic_pii_cleaned.csv')
    cleaner.save_audit_report(audit_report, 'output/nordic_pii_audit.json')
    
    # Generate reports
    print("\n📊 Generating explainability reports...")
    cleaner.save_simple_report(audit_report, 'output/nordic_pii_simple_report.json', 'Nordic Healthcare Dataset')
    cleaner.save_detailed_report(audit_report, 'output/nordic_pii_detailed_report.json', 'Nordic Healthcare Dataset')
    
    print("\n✅ Nordic-specific entities detected:")
    print("  ✓ FI_PERSONAL_ID (Finnish Henkilötunnus)")
    print("  ✓ SE_PERSONAL_ID (Swedish Personnummer)")
    print("  ✓ NO_PERSONAL_ID (Norwegian Fødselsnummer)")
    print("  ✓ DK_PERSONAL_ID (Danish CPR-nummer)")
    
    return cleaned_df, audit_report


def test_interactive_mode():
    """Test interactive mode (requires user input)"""
    print("\n" + "="*70)
    print("TEST 5: Interactive Mode (Manual Decisions)")
    print("="*70)
    
    # Create ambiguous test data
    test_data = pd.DataFrame({
        'id': [1, 2, 3],
        'description': [
            'Customer from Paris contacted us',  # Paris = location or name?
            'Spoke with Jordan about the account',  # Jordan = location or name?
            'Meeting scheduled for March 15th'  # Date
        ],
        'value': [100, 200, 300]
    })
    
    print(f"\n✓ Created dataset with ambiguous PII:")
    print(test_data)
    
    print("\n⚠️  This test requires user input for ambiguous cases.")
    print("    You'll be prompted to choose anonymization strategies.")
    
    proceed = input("\nProceed with interactive test? (y/n): ").strip().lower()
    
    if proceed == 'y':
        cleaner = DataCleaner(test_data)
        cleaned_df, audit_report = cleaner.clean(
            risky_features=None,
            interactive=True,  # Enable interactive prompts
            scan_all_cells=True
        )
        
        print("\n🔒 Cleaned dataset:")
        print(cleaned_df)
        
        cleaner.print_audit_summary(audit_report)
    else:
        print("  Skipped interactive test.")


def demonstrate_integration_with_analysis():
    """Demonstrate how cleaning integrates with AI governance pipeline"""
    print("\n" + "="*70)
    print("TEST 6: Integration Demo (Cleaning → Analysis Workflow)")
    print("="*70)
    
    # Load data
    df = pd.read_csv('Datasets/loan_data.csv')
    
    print("\n📊 Workflow:")
    print("  1. Original dataset → Risk Analysis")
    print("  2. Risk Analysis → Identifies risky features")
    print("  3. Risky features → Data Cleaning (this step)")
    print("  4. Cleaned dataset → Re-run Analysis (optional)")
    
    # Simulate risky features from analysis
    simulated_risky_features = ['person_education', 'loan_intent']
    
    print(f"\n⚠️  Step 2 Output (simulated): Risky features = {simulated_risky_features}")
    
    # Step 3: Clean data
    print("\n🔒 Step 3: Cleaning risky features...")
    cleaner = DataCleaner(df)
    cleaned_df, audit_report = cleaner.clean(
        risky_features=simulated_risky_features,
        interactive=False,
        scan_all_cells=False
    )
    
    # Save both datasets
    os.makedirs('output', exist_ok=True)
    df.to_csv('output/loan_data_original.csv', index=False)
    cleaner.save_cleaned_data(cleaned_df, 'output/loan_data_cleaned.csv')
    cleaner.save_audit_report(audit_report, 'output/cleaning_audit.json')
    
    print("\n💾 Saved files:")
    print("  - output/loan_data_original.csv (original)")
    print("  - output/loan_data_cleaned.csv (cleaned)")
    print("  - output/cleaning_audit.json (audit report)")
    
    print("\n📈 Step 4: User can now choose which dataset to analyze:")
    print("  Option A: Analyze cleaned dataset (privacy-compliant)")
    print("  Option B: Analyze original dataset (for comparison)")
    print("  Option C: Analyze both and compare results")
    
    cleaner.print_audit_summary(audit_report)


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 DATA CLEANING MODULE - TEST SUITE")
    print("   General PII + Nordic-Specific PII Detection")
    print("="*70)
    
    print("\nAvailable tests:")
    print("  1. Basic PII detection on loan dataset")
    print("  2. Cleaning with pre-flagged risky features")
    print("  3. General PII detection (US/International) + Reports")
    print("  4. Nordic PII detection (FI, SE, NO, DK) + Reports")
    print("  5. Interactive mode (requires user input)")
    print("  6. Integration workflow demonstration")
    print("  7. Run all non-interactive tests")
    print("  8. Run Nordic + General PII tests only")
    
    choice = input("\nSelect test (1-8): ").strip()
    
    if choice == '1':
        test_basic_cleaning()
    elif choice == '2':
        test_with_risky_features()
    elif choice == '3':
        test_with_synthetic_pii()
    elif choice == '4':
        test_nordic_pii()
    elif choice == '5':
        test_interactive_mode()
    elif choice == '6':
        demonstrate_integration_with_analysis()
    elif choice == '7':
        print("\n🏃 Running all non-interactive tests...\n")
        test_basic_cleaning()
        test_with_risky_features()
        test_with_synthetic_pii()
        test_nordic_pii()
        demonstrate_integration_with_analysis()
        print("\n✅ All tests completed!")
    elif choice == '8':
        print("\n🏃 Running PII detection tests with report generation...\n")
        test_with_synthetic_pii()
        test_nordic_pii()
        print("\n" + "="*70)
        print("✅ PII TESTS COMPLETED!")
        print("="*70)
        print("\n📂 Generated files in output/:")
        print("  General PII:")
        print("    - general_pii_cleaned.csv")
        print("    - general_pii_audit.json")
        print("    - general_pii_simple_report.json")
        print("    - general_pii_detailed_report.json")
        print("\n  Nordic PII:")
        print("    - nordic_pii_cleaned.csv")
        print("    - nordic_pii_audit.json")
        print("    - nordic_pii_simple_report.json")
        print("    - nordic_pii_detailed_report.json")
        print("\n💡 Review the simple reports for executive summaries")
        print("💡 Review the detailed reports for compliance documentation")
    else:
        print("Invalid choice. Run: python test_cleaning.py")


if __name__ == '__main__':
    main()
