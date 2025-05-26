# core/presence_logs.py
import csv
from datetime import datetime

class PresenceTracker:
    def __init__(self):
        self.records = {}  # {name: {'first_seen': ts, 'last_seen': ts}}
        
    def update_presence(self, name):
        now = datetime.now()
        if name not in self.records:
            self.records[name] = {
                'first_seen': now,
                'last_seen': now,
                'sessions': []
            }
        else:
            last = self.records[name]['last_seen']
            if (now - last).seconds > 300:  # 5 min gap = new session
                self.records[name]['sessions'].append(
                    (last, now)
                )
            self.records[name]['last_seen'] = now
    
    def save_logs(self):
        with open('logs/presence.csv', 'a') as f:
            writer = csv.writer(f)
            for name, data in self.records.items():
                writer.writerow([
                    name,
                    data['first_seen'].isoformat(),
                    data['last_seen'].isoformat(),
                    len(data['sessions'])
                ])