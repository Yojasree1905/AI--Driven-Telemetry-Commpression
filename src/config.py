"""Central configuration: paths, feature schema, attack taxonomy, engine defaults.

Every other module imports its constants from here so the training pipeline, the
inference engine and the UI can never drift out of sync.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data' / 'raw'
MODEL_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
LEGACY_MODEL_DIR = BASE_DIR / 'notebooks' / 'models'

TRAIN_CSV = DATA_DIR / 'UNSW_NB15_training-set.csv'
TEST_CSV = DATA_DIR / 'UNSW_NB15_testing-set.csv'

PREPROCESSOR_PATH = MODEL_DIR / 'preprocessor.joblib'
DETECTOR_PATH = MODEL_DIR / 'detector_xgb.joblib'
CLASSIFIER_PATH = MODEL_DIR / 'classifier_xgb.joblib'
NOVELTY_PATH = MODEL_DIR / 'novelty_iforest.joblib'
RF_PATH = MODEL_DIR / 'detector_rf.joblib'
ENGINE_CONFIG_PATH = MODEL_DIR / 'engine_config.json'
METRICS_PATH = RESULTS_DIR / 'metrics.json'
MODEL_CARD_PATH = RESULTS_DIR / 'model_card.md'

CATEGORICAL_COLS = ['proto', 'service', 'state']

NUMERICAL_COLS = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
    'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
    'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean',
    'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl',
    'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
    'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm',
    'ct_srv_dst', 'is_sm_ips_ports',
]

ALL_FEATURES = CATEGORICAL_COLS + NUMERICAL_COLS

# Columns that must never reach a model (identifiers / targets).
NON_FEATURE_COLS = ['id', 'label', 'attack_cat']

# Telemetry columns the simulator emits alongside the 42 model features. These
# are what a real NetFlow/Zeek sensor records but the UNSW-NB15 corpus strips.
TELEMETRY_COLS = [
    'event_time', 'flow_id', 'sensor', 'direction', 'src_ip', 'src_port',
    'dst_ip', 'dst_port', 'src_zone', 'dst_zone',
]
GROUND_TRUTH_COLS = ['gt_label', 'gt_attack_cat']

# ---------------------------------------------------------------------------
# Attack taxonomy
# ---------------------------------------------------------------------------
BENIGN_CLASS = 'Normal'

ATTACK_CLASSES = [
    'Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers',
    'Generic', 'Reconnaissance', 'Shellcode', 'Worms',
]

# Several UNSW-NB15 families overlap almost completely at the flow level
# (Analysis vs Reconnaissance, Backdoor vs Shellcode, DoS vs Exploits). Grouping
# them gives an honest secondary metric: "did we at least get the family right?"
FAMILY_GROUPS = {
    'Analysis': 'Probe & Analysis',
    'Reconnaissance': 'Probe & Analysis',
    'Backdoor': 'Backdoor & Shellcode',
    'Shellcode': 'Backdoor & Shellcode',
    'DoS': 'DoS & Exploits',
    'Exploits': 'DoS & Exploits',
    'Fuzzers': 'Fuzzing & Protocol Abuse',
    'Generic': 'Generic Cryptographic',
    'Worms': 'Worms & Self-Propagation',
}

# Analyst-facing descriptions surfaced next to a predicted attack type.
ATTACK_PLAYBOOK = {
    'Analysis': {
        'summary': 'Port/mail/web scanning and traffic analysis used to map the estate before an intrusion.',
        'action': 'Correlate the source against other scan hits, rate-limit at the perimeter, open a recon ticket.',
        'severity': 'Medium',
    },
    'Backdoor': {
        'summary': 'Covert channel bypassing authentication, typically a persistent implant beaconing outbound.',
        'action': 'Isolate the internal host, capture memory, hunt for the same beacon interval estate-wide.',
        'severity': 'Critical',
    },
    'DoS': {
        'summary': 'Resource-exhaustion traffic aimed at making a service unavailable.',
        'action': 'Engage upstream scrubbing, apply connection rate limits on the destination VIP.',
        'severity': 'High',
    },
    'Exploits': {
        'summary': 'Traffic carrying a known vulnerability trigger against a service or OS.',
        'action': 'Identify the target CVE from payload context, verify patch level, block the source.',
        'severity': 'Critical',
    },
    'Fuzzers': {
        'summary': 'Malformed or randomised protocol input probing for crash conditions.',
        'action': 'Confirm the target service is stable, capture the malformed payload for the app team.',
        'severity': 'Medium',
    },
    'Generic': {
        'summary': 'Block-cipher / cryptographic attacks that work against any implementation of a scheme.',
        'action': 'Review the key schedule and cipher configuration on the destination service.',
        'severity': 'High',
    },
    'Reconnaissance': {
        'summary': 'Active enumeration of hosts, ports and services ahead of an attack.',
        'action': 'Block the scanning source, verify no follow-on connection succeeded.',
        'severity': 'Medium',
    },
    'Shellcode': {
        'summary': 'Payload delivering executable machine code to hijack a process.',
        'action': 'Treat as confirmed compromise attempt: isolate, snapshot, escalate to IR.',
        'severity': 'Critical',
    },
    'Worms': {
        'summary': 'Self-replicating traffic spreading laterally without operator interaction.',
        'action': 'Contain the VLAN immediately, this spreads faster than manual response.',
        'severity': 'Critical',
    },
    'Unknown': {
        'summary': 'Flow is structurally unlike anything in the training baseline and matches no known family.',
        'action': 'Route to threat hunting: full packet capture, retain the flow, review against fresh intel.',
        'severity': 'High',
    },
}

# ---------------------------------------------------------------------------
# Engine defaults (overridden by models/engine_config.json after training)
# ---------------------------------------------------------------------------
# Share of benign validation traffic we are willing to alert on. This sets the
# detector threshold and is prior-independent, so it survives the shift between
# the training mix and whatever the sensor actually sees.
DEFAULT_ALERT_BUDGET = 0.02

# Novelty score above this validation-normal quantile is treated as structurally
# unfamiliar traffic.
DEFAULT_NOVELTY_QUANTILE = 0.99

# A flow is only labelled a zero-day candidate when the attack-family model is
# also unsure; above this confidence we trust the named family instead.
DEFAULT_FAMILY_CONFIDENCE_FLOOR = 0.55

STATUS_NORMAL = 'Normal'
STATUS_ATTACK = 'Attack'
STATUS_ZERO_DAY = 'Zero-Day Candidate'

SEVERITY_ORDER = ['Info', 'Low', 'Medium', 'High', 'Critical']


def family_group(attack_cat: str) -> str:
    """Collapse a fine-grained family into its overlapping super-family."""
    return FAMILY_GROUPS.get(attack_cat, attack_cat)
