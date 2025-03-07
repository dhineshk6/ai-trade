import pandas as pd
import numpy as np
from datetime import datetime
import os
import re
from typing import Dict, List, Optional

class LogAnalyzer:
    def __init__(self):
        self.current_time = datetime(2025, 3, 6, 23, 57, 8)
        self.current_user = 'dhineshk6'
        self.log_files = {
            'trading': 'logs/trading.log',
            'errors': 'logs/errors.log',
            'performance': 'logs/performance.log'
        }

    def analyze_logs(self, log_type: str, start_time: Optional[datetime] = None) -> Dict:
        """Analyze logs of specified type"""
        try:
            if log_type not in self.log_files:
                raise ValueError(f"Invalid log type: {log_type}")

            log_file = self.log_files[log_type]
            if not os.path.exists(log_file):
                return {'error': f"Log file not found: {log_file}"}

            # Read and parse logs
            logs = self._read_logs(log_file)
            if not logs:
                return {'error': "No logs found"}

            # Filter by start time if provided
            if start_time:
                logs = [log for log in logs if log['timestamp'] >= start_time]

            # Analyze based on log type
            if log_type == 'trading':
                return self._analyze_trading_logs(logs)
            elif log_type == 'errors':
                return self._analyze_error_logs(logs)
            elif log_type == 'performance':
                return self._analyze_performance_logs(logs)

        except Exception as e:
            return {'error': f"Error analyzing logs: {str(e)}"}

    def _read_logs(self, log_file: str) -> List[Dict]:
        """Read and parse log file"""
        logs = []
        log_pattern = re.compile(
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| '
            r'(\w+) \| (\w+) \| User: (\w+) \| (.*)'
        )

        with open(log_file, 'r') as f:
            for line in f:
                match = log_pattern.match(line)
                if match:
                    timestamp_str, level, logger_name, user, message = match.groups()
                    logs.append({
                        'timestamp': datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S'),
                        'level': level,
                        'logger': logger_name,
                        'user': user,
                        'message': message.strip()
                    })

        return logs

    def _analyze_trading_logs(self, logs: List[Dict]) -> Dict:
        """Analyze trading activity logs"""
        trades = []
        signals = []
        
        for log in logs:
            msg = log['message'].lower()
            if 'trade' in msg:
                trades.append(log)
            elif 'signal' in msg:
                signals.append(log)

        return {
            'total_trades': len(trades),
            'total_signals': len(signals),
            'last_trade_time': trades[-1]['timestamp'] if trades else None,
            'last_signal_time': signals[-1]['timestamp'] if signals else None,
            'trade_frequency': self._calculate_frequency(trades),
            'signal_frequency': self._calculate_frequency(signals),
            'timestamp': self.current_time,
            'user': self.current_user
        }

    def _analyze_error_logs(self, logs: List[Dict]) -> Dict:
        """Analyze error logs"""
        error_types = {}
        for log in logs:
            error_msg = log['message']
            error_type = error_msg.split(':')[0] if ':' in error_msg else 'Unknown'
            error_types[error_type] = error_types.get(error_type, 0) + 1

        return {
            'total_errors': len(logs),
            'error_types': error_types,
            'error_frequency': self._calculate_frequency(logs),
            'last_error_time': logs[-1]['timestamp'] if logs else None,
            'timestamp': self.current_time,
            'user': self.current_user
        }

    def _analyze_performance_logs(self, logs: List[Dict]) -> Dict:
        """Analyze performance monitoring logs"""
        metrics = {
            'execution_times': [],
            'memory_usage': [],
            'api_latency': []
        }

        for log in logs:
            msg = log['message'].lower()
            if 'execution time' in msg:
                time_value = self._extract_numeric_value(msg)
                if time_value:
                    metrics['execution_times'].append(time_value)
            elif 'memory usage' in msg:
                memory_value = self._extract_numeric_value(msg)
                if memory_value:
                    metrics['memory_usage'].append(memory_value)
            elif 'api latency' in msg:
                latency_value = self._extract_numeric_value(msg)
                if latency_value:
                    metrics['api_latency'].append(latency_value)

        return {
            'average_execution_time': np.mean(metrics['execution_times']) if metrics['execution_times'] else None,
            'average_memory_usage': np.mean(metrics['memory_usage']) if metrics['memory_usage'] else None,
            'average_api_latency': np.mean(metrics['api_latency']) if metrics['api_latency'] else None,
            'timestamp': self.current_time,
            'user': self.current_user
        }

    def _calculate_frequency(self, logs: List[Dict]) -> float:
        """Calculate frequency of events (events per hour)"""
        if not logs or len(logs) < 2:
            return 0.0

        time_range = (logs[-1]['timestamp'] - logs[0]['timestamp']).total_seconds() / 3600
        return len(logs) / time_range if time_range > 0 else 0.0

    def _extract_numeric_value(self, message: str) -> Optional[float]:
        """Extract numeric value from log message"""
        match = re.search(r'[\d.]+', message)
        return float(match.group()) if match else None

    def generate_report(self, start_time: Optional[datetime] = None) -> Dict:
        """Generate comprehensive log analysis report"""
        return {
            'trading_analysis': self.analyze_logs('trading', start_time),
            'error_analysis': self.analyze_logs('errors', start_time),
            'performance_analysis': self.analyze_logs('performance', start_time),
            'timestamp': self.current_time,
            'user': self.current_user
        }