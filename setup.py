import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def create_project_structure():
    """Create the complete project structure with all necessary files"""
    
    # Create directories
    directories = [
        'backtest',
        'config',
        'data',
        'models',
        'strategies',
        'utils',
        'logs',
        'tests'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        (Path(directory) / '__init__.py').touch()
        print(f"Created directory: {directory}")

def setup_environment():
    """Setup virtual environment and install requirements"""
    try:
        # Create virtual environment
        if not Path('venv').exists():
            subprocess.run([sys.executable, '-m', 'venv', 'venv'])
            print("Created virtual environment")

        # Create requirements.txt
        requirements = """ccxt==4.0.0
pandas==1.5.3
numpy==1.21.6
sqlalchemy==1.4.46
scikit-learn==1.0.2
tensorflow==2.11.0
ta==0.10.2
python-dotenv==1.0.0
matplotlib==3.5.3
scipy==1.10.1
seaborn==0.12.2
requests==2.31.0"""
        
        Path('requirements.txt').write_text(requirements)
        print("Created requirements.txt")

        # Create .env file
        env_content = """KUCOIN_API_KEY=your_api_key_here
KUCOIN_API_SECRET=your_secret_here
KUCOIN_API_PASSPHRASE=your_passphrase_here
DATABASE_URL=sqlite:///trading_bot.db"""
        
        Path('.env').write_text(env_content)
        print("Created .env file")

    except Exception as e:
        print(f"Error during environment setup: {e}")
        sys.exit(1)

def main():
    print("Setting up KuCoin AI Trading Bot...")
    
    # Create project structure
    create_project_structure()
    
    # Setup environment
    setup_environment()
    
    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print("1. Update your KuCoin API credentials in .env")
    print("2. Install requirements:")
    print("   pip install -r requirements.txt")
    print("3. Run the bot:")
    print("   python main.py")

if __name__ == "__main__":
    main()