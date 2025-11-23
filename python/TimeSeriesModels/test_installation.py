"""
Test script to verify installation and basic functionality.
"""

import sys
import importlib

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    
    packages = [
        'numpy',
        'pandas', 
        'matplotlib',
        'seaborn',
        'sklearn',
        'tensorflow',
        'yfinance',
        'scipy',
        'statsmodels'
    ]
    
    failed_imports = []
    
    for package in packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nFailed to import: {failed_imports}")
        print("Please install missing packages using: pip install -r requirements.txt")
        return False
    else:
        print("\nAll packages imported successfully!")
        return True

def test_custom_modules():
    """Test that our custom modules can be imported."""
    print("\nTesting custom module imports...")
    
    modules = [
        'data_fetcher',
        'ar1_model', 
        'lstm_model',
        'backtesting'
    ]
    
    failed_imports = []
    
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\nFailed to import custom modules: {failed_imports}")
        return False
    else:
        print("\nAll custom modules imported successfully!")
        return True

def test_basic_functionality():
    """Test basic functionality of our models."""
    print("\nTesting basic model functionality...")
    
    try:
        import numpy as np
        from ar1_model import AR1Model
        from lstm_model import LSTMModel
        
        # Generate sample data
        np.random.seed(42)
        n = 100
        data = np.random.normal(0, 1, n).cumsum() + 1.5  # Random walk starting at 1.5
        
        # Test AR1 model
        print("Testing AR1 model...")
        ar1_model = AR1Model()
        ar1_model.fit(data, method='mle')
        ar1_pred = ar1_model.predict(data[-1], steps=5)
        print(f"  AR1 prediction: {ar1_pred}")
        
        # Test LSTM model (with reduced complexity for testing)
        print("Testing LSTM model...")
        lstm_model = LSTMModel(sequence_length=4, hidden_units=10, dropout_rate=0.1)
        lstm_model.fit(data, epochs=10, batch_size=16, verbose=0)
        lstm_pred = lstm_model.predict(data, steps=5)
        print(f"  LSTM prediction: {lstm_pred}")
        
        print("✓ Basic model functionality test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_data_fetcher():
    """Test data fetcher functionality."""
    print("\nTesting data fetcher...")
    
    try:
        from data_fetcher import ExchangeRateDataFetcher
        
        # Test data fetcher initialization
        fetcher = ExchangeRateDataFetcher()
        print("✓ Data fetcher initialized successfully")
        
        # Note: We won't actually fetch data in the test to avoid network dependencies
        print("✓ Data fetcher test passed (network test skipped)")
        return True
        
    except Exception as e:
        print(f"✗ Data fetcher test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*50)
    print("INSTALLATION AND FUNCTIONALITY TEST")
    print("="*50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Custom Modules", test_custom_modules),
        ("Basic Functionality", test_basic_functionality),
        ("Data Fetcher", test_data_fetcher)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your installation is ready.")
        print("\nYou can now run the main analysis with:")
        print("python main.py")
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Check Python version (3.7+ required)")
        print("3. For TensorFlow issues, try: pip install tensorflow-cpu")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
