import subprocess
import os

def run_streamlit():
    # Recieve route of the script
    script_path = os.path.join(os.path.dirname(__file__), "ASM Calculator.py")

    # Execute the App
    subprocess.run(["streamlit", "run", script_path])

if __name__ == "__main__":
    run_streamlit()