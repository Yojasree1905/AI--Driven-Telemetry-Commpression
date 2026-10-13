"""
Database & Cloud Telemetry Vault Module with Peer Connection Requests.

Supports:
1. Multi-User Authentication (Username, Full Name, Email, Last Active)
2. Peer Connection Requests ('PENDING', 'ACCEPTED', 'REJECTED')
3. Upload and Transmissions Ledger (Compressed CSV, PDF, or Dual Payloads)
4. Bit-Exact Cryptographic Hash Verification (SHA-256) & Quality Loss Auditing
5. Firebase Realtime / Firestore Cloud Sync (Optional with automatic local vault fallback)
"""
import sqlite3
import json
import hashlib
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

# IST Timezone (+05:30)
IST = timezone(timedelta(hours=5, minutes=30))

def get_ist_now_str() -> str:
    """Returns formatted timestamp in IST (Indian Standard Time)."""
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

# Local Vault Storage
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
DB_PATH = PROJECT_ROOT / "data" / "telemetry_vault.db"
VAULT_STORAGE = PROJECT_ROOT / "results" / "vault"
CREDENTIALS_PATH = PROJECT_ROOT / "credentials" / "serviceAccountKey.json"

# Firebase Cloud Client (Optional)
_FIREBASE_INITIALIZED = False
_FIRESTORE_DB = None

def get_firebase_status() -> Tuple[bool, str]:
    """Checks if Firebase Admin SDK is authenticated with a valid service account."""
    global _FIREBASE_INITIALIZED, _FIRESTORE_DB
    if _FIREBASE_INITIALIZED and _FIRESTORE_DB is not None:
        return True, "Firebase Cloud Sync Active"
    
    if CREDENTIALS_PATH.exists():
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore
            if not firebase_admin._apps:
                cred = credentials.Certificate(str(CREDENTIALS_PATH))
                firebase_admin.initialize_app(cred)
            _FIRESTORE_DB = firestore.client()
            _FIREBASE_INITIALIZED = True
            return True, "Firebase Cloud Connected"
        except Exception as e:
            return False, f"Firebase config error: {e}"
    return False, "Using Local Network Vault (Place serviceAccountKey.json in credentials/ for Firebase Cloud)"

def get_db_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    VAULT_STORAGE.mkdir(parents=True, exist_ok=True)
    with get_db_connection() as conn:
        # 1. Users Table
        conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            role TEXT DEFAULT 'User', -- 'User' or 'Admin'
            created_at TEXT NOT NULL,
            last_active TEXT NOT NULL
        )
        ''')
        # Check if role column exists in users table
        u_cols = [r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
        if 'role' not in u_cols:
            conn.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'User'")

        # 2. Peer Connections & Requests Table
        conn.execute('''
        CREATE TABLE IF NOT EXISTS peer_connections (
            connection_id TEXT PRIMARY KEY,
            sender_username TEXT NOT NULL,
            target_username TEXT NOT NULL,
            status TEXT NOT NULL, -- 'PENDING', 'ACCEPTED', 'REJECTED'
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        ''')

        # Check and migrate transmissions table if outdated schema exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='transmissions'")
        if cursor.fetchone():
            existing_cols = [r[1] for r in conn.execute("PRAGMA table_info(transmissions)").fetchall()]
            if 'payload_type' not in existing_cols:
                # Legacy table detected without payload_type - recreate transmissions table safely
                conn.execute("DROP TABLE IF EXISTS audit_receipts")
                conn.execute("DROP TABLE IF EXISTS transmissions")

        # 3. Transmissions Table
        conn.execute('''
        CREATE TABLE IF NOT EXISTS transmissions (
            transmission_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            sender_username TEXT NOT NULL,
            recipient_username TEXT NOT NULL,
            payload_type TEXT NOT NULL, -- 'CSV', 'PDF', 'DUAL'
            raw_filename TEXT NOT NULL,
            raw_size_bytes INTEGER NOT NULL,
            comp_size_bytes INTEGER NOT NULL,
            comp_algo TEXT NOT NULL,
            compression_ratio REAL NOT NULL,
            space_saving_pct REAL NOT NULL,
            sha256_original TEXT NOT NULL,
            blob_path TEXT NOT NULL,
            pdf_raw_size_bytes INTEGER DEFAULT 0,
            pdf_comp_size_bytes INTEGER DEFAULT 0,
            pdf_sha256_original TEXT DEFAULT '',
            pdf_blob_path TEXT DEFAULT '',
            top_3_json TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            status TEXT NOT NULL
        )
        ''')

        # 4. Audit Receipts Table
        conn.execute('''
        CREATE TABLE IF NOT EXISTS audit_receipts (
            receipt_id TEXT PRIMARY KEY,
            transmission_id TEXT NOT NULL,
            auditor_username TEXT NOT NULL,
            audit_timestamp TEXT NOT NULL,
            decompressed_sha256 TEXT NOT NULL,
            is_bit_exact INTEGER NOT NULL,
            mse REAL NOT NULL,
            psnr REAL NOT NULL,
            data_loss_percentage REAL NOT NULL,
            notes TEXT,
            FOREIGN KEY (transmission_id) REFERENCES transmissions(transmission_id)
        )
        ''')
        conn.commit()

def compute_sha256(data: bytes) -> str:
    """Computes cryptographic SHA-256 hash for bit-exact verification."""
    return hashlib.sha256(data).hexdigest()

# -----------------------------------------------------------------------------
# USER AUTHENTICATION
# -----------------------------------------------------------------------------
def register_or_login_user(name: str, email: str, role: str = "User") -> Dict[str, Any]:
    """Registers or logs in a user given their Name, Email, and selected Role."""
    init_db()
    name = name.strip()
    email = email.strip().lower()
    role = role.strip() if role else "User"
    
    username = email.split('@')[0].replace('.', '_').replace('-', '_')
    if not username:
        username = name.lower().replace(' ', '_')

    now_str = get_ist_now_str()

    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ? OR email = ?", (username, email)).fetchone()
        if row:
            actual_username = row['username']
            conn.execute("UPDATE users SET last_active = ?, name = ?, role = ? WHERE username = ?", (now_str, name, role, actual_username))
            conn.commit()
            user_data = dict(row)
            user_data['last_active'] = now_str
            user_data['name'] = name
            user_data['role'] = role
        else:
            conn.execute(
                "INSERT INTO users (username, name, email, role, created_at, last_active) VALUES (?, ?, ?, ?, ?, ?)",
                (username, name, email, role, now_str, now_str)
            )
            conn.commit()
            user_data = {
                "username": username,
                "name": name,
                "email": email,
                "role": role,
                "created_at": now_str,
                "last_active": now_str
            }

    # Sync to Firebase if active
    is_fb, _ = get_firebase_status()
    if is_fb and _FIRESTORE_DB:
        try:
            _FIRESTORE_DB.collection("users").document(user_data["username"]).set(user_data, merge=True)
        except Exception:
            pass

    return user_data

def list_all_users() -> List[Dict[str, Any]]:
    """Returns list of all registered users."""
    init_db()
    with get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY last_active DESC").fetchall()
        return [dict(r) for r in rows]

def list_admins() -> List[Dict[str, Any]]:
    """Returns list of all registered Admin users."""
    init_db()
    with get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM users WHERE role = 'Admin' ORDER BY last_active DESC").fetchall()
        return [dict(r) for r in rows]

def list_regular_users() -> List[Dict[str, Any]]:
    """Returns list of all registered regular Client users."""
    init_db()
    with get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM users WHERE role = 'User' ORDER BY last_active DESC").fetchall()
        return [dict(r) for r in rows]

# -----------------------------------------------------------------------------
# PEER CONNECTION REQUESTS & ACCEPTANCE
# -----------------------------------------------------------------------------
def send_connection_request(sender_username: str, target_username: str) -> str:
    """Sends a connection request to another user."""
    init_db()
    conn_id = f"CONN_{min(sender_username, target_username)}_{max(sender_username, target_username)}"
    now_str = get_ist_now_str()
    with get_db_connection() as conn:
        # Check if exists
        row = conn.execute("SELECT * FROM peer_connections WHERE connection_id = ?", (conn_id,)).fetchone()
        if row:
            if row['status'] == 'ACCEPTED':
                return "ALREADY_CONNECTED"
            elif row['status'] == 'PENDING':
                return "ALREADY_PENDING"
        
        conn.execute('''
        INSERT OR REPLACE INTO peer_connections (connection_id, sender_username, target_username, status, created_at, updated_at)
        VALUES (?, ?, ?, 'PENDING', ?, ?)
        ''', (conn_id, sender_username, target_username, now_str, now_str))
        conn.commit()
    return "REQUEST_SENT"

def accept_connection_request(sender_username: str, target_username: str):
    """Accepts an incoming peer connection request."""
    init_db()
    conn_id = f"CONN_{min(sender_username, target_username)}_{max(sender_username, target_username)}"
    now_str = get_ist_now_str()
    with get_db_connection() as conn:
        conn.execute("UPDATE peer_connections SET status = 'ACCEPTED', updated_at = ? WHERE connection_id = ?", (now_str, conn_id))
        conn.commit()

def reject_connection_request(sender_username: str, target_username: str):
    """Rejects an incoming peer connection request."""
    init_db()
    conn_id = f"CONN_{min(sender_username, target_username)}_{max(sender_username, target_username)}"
    now_str = get_ist_now_str()
    with get_db_connection() as conn:
        conn.execute("UPDATE peer_connections SET status = 'REJECTED', updated_at = ? WHERE connection_id = ?", (now_str, conn_id))
        conn.commit()

def get_connected_peers(username: str) -> List[str]:
    """Returns list of usernames that are mutually connected (ACCEPTED)."""
    init_db()
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT sender_username, target_username FROM peer_connections WHERE (sender_username = ? OR target_username = ?) AND status = 'ACCEPTED'",
            (username, username)
        ).fetchall()
        peers = set()
        for r in rows:
            if r['sender_username'] == username:
                peers.add(r['target_username'])
            else:
                peers.add(r['sender_username'])
        return sorted(list(peers))

def get_pending_incoming_requests(username: str) -> List[Dict[str, Any]]:
    """Returns pending connection requests waiting for this user's acceptance."""
    init_db()
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM peer_connections WHERE target_username = ? AND status = 'PENDING' ORDER BY created_at DESC",
            (username,)
        ).fetchall()
        return [dict(r) for r in rows]

# -----------------------------------------------------------------------------
# TRANSMISSION DISPATCH & INBOX
# -----------------------------------------------------------------------------
def dispatch_transmission_package(
    transmission_id: str,
    sender_username: str,
    recipient_username: str,
    payload_type: str,
    raw_filename: str,
    raw_bytes: bytes,
    compressed_bytes: bytes,
    algo_used: str,
    compression_ratio: float,
    space_saving_pct: float,
    top_3_list: List[Dict[str, Any]],
    metrics_dict: Dict[str, Any],
    pdf_raw_bytes: Optional[bytes] = None,
    pdf_comp_bytes: Optional[bytes] = None
) -> str:
    """Dispatches a compressed package to a connected peer."""
    init_db()
    safe_trans = transmission_id.replace(' ', '_').replace('/', '_')
    blob_path = VAULT_STORAGE / f"{safe_trans}_payload.bin"
    blob_path.write_bytes(compressed_bytes)

    sha256_orig = compute_sha256(raw_bytes)

    pdf_blob_path_str = ""
    pdf_sha = ""
    pdf_raw_len = 0
    pdf_comp_len = 0
    if pdf_raw_bytes and pdf_comp_bytes:
        pdf_path = VAULT_STORAGE / f"{safe_trans}_audit.pdf.bin"
        pdf_path.write_bytes(pdf_comp_bytes)
        pdf_blob_path_str = str(pdf_path)
        pdf_sha = compute_sha256(pdf_raw_bytes)
        pdf_raw_len = len(pdf_raw_bytes)
        pdf_comp_len = len(pdf_comp_bytes)

    now_str = get_ist_now_str()

    trans_record = {
        "transmission_id": transmission_id,
        "timestamp": now_str,
        "sender_username": sender_username,
        "recipient_username": recipient_username,
        "payload_type": payload_type,
        "raw_filename": raw_filename,
        "raw_size_bytes": len(raw_bytes),
        "comp_size_bytes": len(compressed_bytes),
        "comp_algo": algo_used,
        "compression_ratio": float(compression_ratio),
        "space_saving_pct": float(space_saving_pct),
        "sha256_original": sha256_orig,
        "blob_path": str(blob_path),
        "pdf_raw_size_bytes": pdf_raw_len,
        "pdf_comp_size_bytes": pdf_comp_len,
        "pdf_sha256_original": pdf_sha,
        "pdf_blob_path": pdf_blob_path_str,
        "top_3_json": json.dumps(top_3_list),
        "metrics_json": json.dumps(metrics_dict),
        "status": "SENT"
    }

    with get_db_connection() as conn:
        conn.execute('''
        INSERT OR REPLACE INTO transmissions (
            transmission_id, timestamp, sender_username, recipient_username,
            payload_type, raw_filename, raw_size_bytes, comp_size_bytes, comp_algo,
            compression_ratio, space_saving_pct, sha256_original, blob_path,
            pdf_raw_size_bytes, pdf_comp_size_bytes, pdf_sha256_original, pdf_blob_path,
            top_3_json, metrics_json, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', tuple(trans_record.values()))
        conn.commit()

    return transmission_id

def get_inbox_transmissions(recipient_username: str) -> List[Dict[str, Any]]:
    """Retrieves all transmissions sent to the specified user."""
    init_db()
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM transmissions WHERE recipient_username = ? ORDER BY timestamp DESC",
            (recipient_username,)
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            try:
                d["metrics"] = json.loads(d["metrics_json"])
            except:
                d["metrics"] = {}
            try:
                d["top_3"] = json.loads(d["top_3_json"])
            except:
                d["top_3"] = []
            results.append(d)
        return results

def get_transmission(transmission_id: str) -> Optional[Dict[str, Any]]:
    """Fetches full transmission record."""
    init_db()
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM transmissions WHERE transmission_id = ?", (transmission_id,)).fetchone()
        if not row:
            return None
        res = dict(row)
        try:
            res["metrics"] = json.loads(res["metrics_json"])
        except:
            res["metrics"] = {}
        try:
            res["top_3"] = json.loads(res["top_3_json"])
        except:
            res["top_3"] = []
        return res

def read_transmission_blob(transmission_id: str) -> bytes:
    """Reads the primary compressed payload blob."""
    rec = get_transmission(transmission_id)
    if not rec:
        raise ValueError("Transmission not found")
    return Path(rec["blob_path"]).read_bytes()

def read_transmission_pdf_blob(transmission_id: str) -> Optional[bytes]:
    """Reads the secondary compressed PDF blob if present."""
    rec = get_transmission(transmission_id)
    if not rec or not rec.get("pdf_blob_path"):
        return None
    p = Path(rec["pdf_blob_path"])
    return p.read_bytes() if p.exists() else None

# -----------------------------------------------------------------------------
# AUDIT RECEIPT & QUALITY LOSS RECORDING
# -----------------------------------------------------------------------------
def record_audit_receipt(
    receipt_id: str,
    transmission_id: str,
    auditor_username: str,
    decompressed_bytes: bytes,
    mse: float = 0.0,
    psnr: float = 999.0,
    notes: str = ""
) -> Dict[str, Any]:
    """Records recipient decompression audit and verifies data integrity."""
    init_db()
    trans = get_transmission(transmission_id)
    if not trans:
        raise ValueError(f"Transmission {transmission_id} not found.")

    decomp_sha = compute_sha256(decompressed_bytes)
    is_exact = (decomp_sha == trans["sha256_original"])
    data_loss_pct = 0.0 if is_exact else (min(100.0, max(0.01, mse * 100.0)))
    now_str = get_ist_now_str()

    receipt = {
        "receipt_id": receipt_id,
        "transmission_id": transmission_id,
        "auditor_username": auditor_username,
        "audit_timestamp": now_str,
        "decompressed_sha256": decomp_sha,
        "is_bit_exact": 1 if is_exact else 0,
        "mse": float(mse),
        "psnr": float(psnr),
        "data_loss_percentage": float(data_loss_pct),
        "notes": notes
    }

    with get_db_connection() as conn:
        conn.execute('''
        INSERT OR REPLACE INTO audit_receipts (
            receipt_id, transmission_id, auditor_username, audit_timestamp,
            decompressed_sha256, is_bit_exact, mse, psnr, data_loss_percentage, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', tuple(receipt.values()))
        conn.execute("UPDATE transmissions SET status = 'RECEIVED_AND_AUDITED' WHERE transmission_id = ?", (transmission_id,))
        conn.commit()

    return receipt
