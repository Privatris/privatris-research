import subprocess
import sys
import os

def run_tests():
    """
    Runs the test suite for the PRIVATRIS framework.
    """
    print("=" * 80)
    print("PRIVATRIS TEST RUNNER")
    print("=" * 80)
    
    # Define tests to run
    tests = [
        ("Component Tests", "test_components.py"),
        # Add other tests here as needed, e.g.,
        # ("GPT-2 Fast Test", "test_gpt2_fast.py"),
    ]
    
    failed_tests = []
    
    for test_name, test_file in tests:
        print(f"\nRunning {test_name} ({test_file})...")
        print("-" * 40)
        
        try:
            # Run the test file as a subprocess
            result = subprocess.run(
                [sys.executable, test_file],
                capture_output=False, # Let output flow to stdout
                text=True,
                check=True
            )
            print(f"\n✅ {test_name} PASSED")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ {test_name} FAILED with exit code {e.returncode}")
            failed_tests.append(test_name)
        except Exception as e:
            print(f"\n❌ {test_name} FAILED with error: {e}")
            failed_tests.append(test_name)
            
    print("\n" + "=" * 80)
    if failed_tests:
        print(f"❌ SUMMARY: {len(failed_tests)} tests failed.")
        for t in failed_tests:
            print(f"  - {t}")
        sys.exit(1)
    else:
        print("✅ SUMMARY: All tests passed successfully.")
        sys.exit(0)

if __name__ == "__main__":
    # Ensure we are in the correct directory (where this script is)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    run_tests()
