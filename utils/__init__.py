from datetime import datetime
from typing import Dict, List, Optional

# Module metadata
__version__ = '1.0.0'
__author__ = 'dhineshk6'
__updated__ = '2025-03-07 00:00:20'

# Current session information
CURRENT_TIME = '2025-03-07 00:00:20'
CURRENT_USER = 'dhineshk6'

class Utilities:
    @staticmethod
    def get_current_session_info() -> Dict:
        """Get current session information"""
        return {
            'timestamp': CURRENT_TIME,
            'user': CURRENT_USER,
            'version': __version__,
            'last_updated': __updated__
        }

    @staticmethod
    def format_timestamp(dt: datetime) -> str:
        """Format datetime to standard UTC string"""
        return dt.strftime('%Y-%m-%d %H:%M:%S')

    @staticmethod
    def parse_timestamp(timestamp_str: str) -> datetime:
        """Parse timestamp string to datetime object"""
        return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

# Export utilities
__all__ = ['Utilities']

# Initialize utilities
utils = Utilities()