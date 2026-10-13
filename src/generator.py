from typing import Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from src.loader import ALL_FEATURES


def _sample_normal_web_traffic(n_samples, random_seed):
    dataset_path = Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'UNSW_NB15_training-set.csv'
    if not dataset_path.exists():
        return None

    dataset = pd.read_csv(dataset_path)
    normal_web = dataset[
        (dataset['label'] == 0)
        & (dataset['proto'] == 'tcp')
        & (dataset['service'].isin(['http', '-']))
        & (dataset['state'] == 'FIN')
    ]
    if normal_web.empty:
        return None

    sampled = normal_web.sample(n=n_samples, replace=n_samples > len(normal_web), random_state=random_seed)
    return sampled[ALL_FEATURES].reset_index(drop=True)

def generate_synthetic_dataset(technique="Mixed Traffic Stream", n_samples=50, random_seed=42):
    """
    Generates dynamic synthetic network traffic datasets using various generation techniques.
    
    Techniques:
    1. 'Mixed Traffic Stream': Realistic blend of Normal (70%), Attack (20%), and Zero-Day Anomaly (10%).
    2. 'DDoS & Volumetric Attacks': High packet rate, high source load, high connection counts.
    3. 'Zero-Day Exfiltration Anomalies': Atypical byte distributions, unusual TTL, microsecond durations.
    4. 'Normal Web Traffic': Standard HTTP/DNS browsing connections.
    """
    np.random.seed(random_seed)

    if technique == 'Normal Web Traffic':
        sampled_normal = _sample_normal_web_traffic(n_samples, random_seed)
        if sampled_normal is not None:
            return sampled_normal

    records = []
    
    protocols = ['tcp', 'udp', 'arp', 'ospf', 'sctp']
    services = ['http', 'dns', 'ftp', 'smtp', 'ssh', 'ssl', '-']
    states = ['FIN', 'INT', 'CON', 'REQ', 'ACC', 'RST']
    
    for i in range(n_samples):
        # Determine scenario type based on technique
        if technique == "Mixed Traffic Stream":
            roll = np.random.rand()
            if roll < 0.70:
                traffic_type = "normal"
            elif roll < 0.90:
                traffic_type = "ddos"
            else:
                traffic_type = "zeroday"
        elif technique == "DDoS & Volumetric Attacks":
            traffic_type = "ddos"
        elif technique == "Zero-Day Exfiltration Anomalies":
            traffic_type = "zeroday"
        else:
            traffic_type = "normal"
            
        if traffic_type == "normal":
            proto = 'tcp'
            service = 'http'
            state = 'FIN'
            dur = round(float(np.random.exponential(scale=0.08) + 0.005), 6)
            spkts = int(np.random.randint(4, 30))
            dpkts = int(np.random.randint(4, 25))
            sbytes = int(spkts * np.random.randint(80, 200))
            dbytes = int(dpkts * np.random.randint(100, 300))
            rate = round(float((spkts + dpkts) / (dur + 0.001)), 2)
            sttl = int(np.random.choice([64, 31, 62]))
            dttl = int(np.random.choice([64, 29, 62]))
            sload = round(float(sbytes * 8 / (dur + 0.001)), 2)
            dload = round(float(dbytes * 8 / (dur + 0.001)), 2)
            ct_srv_src = int(np.random.randint(1, 4))
            ct_state_ttl = 1
            ct_dst_ltm = int(np.random.randint(1, 4))
            ct_src_dport_ltm = 1
            ct_dst_sport_ltm = 1
            ct_dst_src_ltm = int(np.random.randint(1, 4))
            
        elif traffic_type == "ddos":
            proto = 'tcp'
            service = np.random.choice(['http', '-'])
            state = np.random.choice(['CON', 'FIN', 'INT'])
            dur = round(float(np.random.uniform(0.3, 2.5)), 6)
            spkts = int(np.random.randint(150, 600))
            dpkts = int(np.random.randint(100, 500))
            sbytes = int(spkts * np.random.randint(300, 800))
            dbytes = int(dpkts * np.random.randint(300, 1200))
            rate = round(float((spkts + dpkts) / (dur + 0.001)), 2)
            sttl = 254
            dttl = 0
            sload = round(float(np.random.uniform(2000000.0, 10000000.0)), 2)
            dload = 0.0
            ct_srv_src = int(np.random.randint(10, 25))
            ct_state_ttl = int(np.random.randint(2, 6))
            ct_dst_ltm = int(np.random.randint(10, 25))
            ct_src_dport_ltm = int(np.random.randint(8, 20))
            ct_dst_sport_ltm = int(np.random.randint(5, 15))
            ct_dst_src_ltm = int(np.random.randint(10, 25))
            
        elif traffic_type == "zeroday":
            proto = 'tcp'
            service = 'http'
            state = 'FIN'
            dur = round(float(np.random.uniform(12.0, 45.0)), 6)
            spkts = int(np.random.randint(2, 4))
            dpkts = int(np.random.randint(1, 3))
            sbytes = int(np.random.randint(12000, 35000))
            dbytes = int(np.random.randint(40, 150))
            rate = round(float((spkts + dpkts) / (dur + 0.001)), 2)
            sttl = 64
            dttl = 62
            sload = round(float(sbytes * 8 / (dur + 0.001)), 2)
            dload = round(float(dbytes * 8 / (dur + 0.001)), 2)
            ct_srv_src = int(np.random.randint(1, 4))
            ct_state_ttl = 1
            ct_dst_ltm = int(np.random.randint(1, 4))
            ct_src_dport_ltm = 1
            ct_dst_sport_ltm = 1
            ct_dst_src_ltm = int(np.random.randint(1, 4))

        row = {
            'dur': dur, 'proto': proto, 'service': service, 'state': state,
            'spkts': spkts, 'dpkts': dpkts, 'sbytes': sbytes, 'dbytes': dbytes,
            'rate': rate, 'sttl': sttl, 'dttl': dttl, 'sload': sload, 'dload': dload,
            'sloss': int(spkts * 0.05), 'dloss': int(dpkts * 0.05),
            'sinpkt': round(float(dur * 1000 / (spkts + 1)), 2),
            'dinpkt': round(float(dur * 1000 / (dpkts + 1)), 2),
            'sjit': round(float(np.random.uniform(5.0, 50.0)), 2),
            'djit': round(float(np.random.uniform(5.0, 50.0)), 2),
            'swin': 255 if proto == 'tcp' else 0,
            'stcpb': int(np.random.randint(100000, 9999999)) if proto == 'tcp' else 0,
            'dtcpb': int(np.random.randint(100000, 9999999)) if proto == 'tcp' else 0,
            'dwin': 255 if proto == 'tcp' else 0,
            'tcprtt': round(float(np.random.uniform(0.001, 0.05)), 4),
            'synack': round(float(np.random.uniform(0.001, 0.02)), 4),
            'ackdat': round(float(np.random.uniform(0.001, 0.02)), 4),
            'smean': int(sbytes / (spkts + 1)),
            'dmean': int(dbytes / (dpkts + 1)),
            'trans_depth': 1 if service == 'http' else 0,
            'response_body_len': int(np.random.choice([0, 120, 1500])) if service == 'http' else 0,
            'ct_srv_src': ct_srv_src, 'ct_state_ttl': ct_state_ttl,
            'ct_dst_ltm': ct_dst_ltm, 'ct_src_dport_ltm': ct_src_dport_ltm,
            'ct_dst_sport_ltm': ct_dst_sport_ltm, 'ct_dst_src_ltm': ct_dst_src_ltm,
            'is_ftp_login': 1 if service == 'ftp' else 0,
            'ct_ftp_cmd': 1 if service == 'ftp' else 0,
            'ct_flw_http_mthd': 1 if service == 'http' else 0,
            'ct_src_ltm': ct_dst_ltm, 'ct_srv_dst': ct_srv_src,
            'is_sm_ips_ports': 0
        }
        records.append(row)
        
    df_gen = pd.DataFrame(records)
    return df_gen[ALL_FEATURES]


def generate_volumetric_telemetry_csv(target_size_mb: float = 5.2) -> Tuple[pd.DataFrame, bytes]:
    """Generates heavy, high-volume realistic network telemetry exceeding bandwidth thresholds.
    
    Produces a dataset with 30,000+ records to simulate an oversized telemetry log
    (typically 5MB - 10MB+) that triggers real-world network transmission limits.
    """
    from src.loader import get_sample_test_data
    base_sample = get_sample_test_data(500)
    base_bytes_len = len(base_sample.to_csv(index=False).encode('utf-8'))
    
    target_bytes = int(target_size_mb * 1024 * 1024)
    multiplier = max(10, int(target_bytes / max(1, base_bytes_len)) + 1)
    
    # Generate volumetric replicated dataframe with randomized micro-jitter on timestamps and packets
    df_volumetric = pd.concat([base_sample] * multiplier, ignore_index=True)
    
    # Micro-randomize packet counts and durations to create realistic high-entropy telemetry
    noise = np.random.uniform(0.98, 1.02, size=len(df_volumetric))
    if 'dur' in df_volumetric.columns:
        df_volumetric['dur'] = (df_volumetric['dur'] * noise).round(5)
    if 'sbytes' in df_volumetric.columns:
        df_volumetric['sbytes'] = (df_volumetric['sbytes'] * noise).astype(int)
    if 'dbytes' in df_volumetric.columns:
        df_volumetric['dbytes'] = (df_volumetric['dbytes'] * noise).astype(int)
        
    raw_csv_bytes = df_volumetric.to_csv(index=False).encode('utf-8')
    return df_volumetric, raw_csv_bytes
