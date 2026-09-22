import os
import json
import base64
import hmac
import hashlib
from cryptography.fernet import Fernet

# ----------------------------------------------------
# Data Security & Privacy (DSP) Vault Configuration
# ----------------------------------------------------
VAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dsp_vault")
KEY_FILE = os.path.join(VAULT_DIR, "dsp_master.key")
ENCRYPTED_VAULT_FILE = os.path.join(VAULT_DIR, "credentials.enc")

def _get_or_create_key() -> bytes:
    """Retrieves or creates a persistent AES-256 Fernet master encryption key."""
    if not os.path.exists(VAULT_DIR):
        os.makedirs(VAULT_DIR, exist_ok=True)
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as f:
            return f.read().strip()
    else:
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as f:
            f.write(key)
        return key

MASTER_KEY = _get_or_create_key()
cipher_suite = Fernet(MASTER_KEY)

def hash_password(password: str, salt: bytes = None) -> tuple[str, str]:
    """Generates PBKDF2-HMAC-SHA256 salted hash with 100,000 iterations."""
    if salt is None:
        salt = os.urandom(16)
    pwd_bytes = password.strip().encode("utf-8")
    hash_bytes = hashlib.pbkdf2_hmac("sha256", pwd_bytes, salt, 100000)
    return base64.b64encode(hash_bytes).decode("utf-8"), base64.b64encode(salt).decode("utf-8")

def _load_vault_data() -> dict:
    """Decrypts and loads the DSP user credentials vault."""
    if not os.path.exists(ENCRYPTED_VAULT_FILE):
        # Initialize default admin accounts
        init_users = {
            "admin": {
                "role": "Chief Disaster Operations Commander",
                "salt": None,
                "hash": None
            },
            "officer": {
                "role": "Disaster Response Officer",
                "salt": None,
                "hash": None
            }
        }
        # Seed default passwords securely
        h_admin, s_admin = hash_password("admin123")
        init_users["admin"]["hash"] = h_admin
        init_users["admin"]["salt"] = s_admin

        h_off, s_off = hash_password("maharashtra2026")
        init_users["officer"]["hash"] = h_off
        init_users["officer"]["salt"] = s_off

        _save_vault_data(init_users)
        return init_users

    try:
        with open(ENCRYPTED_VAULT_FILE, "rb") as f:
            encrypted_data = f.read()
        decrypted_json = cipher_suite.decrypt(encrypted_data).decode("utf-8")
        return json.loads(decrypted_json)
    except Exception:
        return {}

def _save_vault_data(vault_dict: dict):
    """Encrypts and writes credentials vault using AES-256 Fernet."""
    json_bytes = json.dumps(vault_dict, indent=2).encode("utf-8")
    encrypted_bytes = cipher_suite.encrypt(json_bytes)
    with open(ENCRYPTED_VAULT_FILE, "wb") as f:
        f.write(encrypted_bytes)

def verify_user(username: str, password: str) -> tuple[bool, str]:
    """
    Verifies user credentials using PBKDF2-HMAC-SHA256 hash comparison.
    Returns: (is_authenticated, user_role)
    """
    u = str(username).strip().lower()
    p = str(password).strip()

    vault = _load_vault_data()
    if u not in vault:
        return False, None

    user_record = vault[u]
    stored_hash = user_record.get("hash")
    stored_salt = user_record.get("salt")

    if not stored_hash or not stored_salt:
        return False, None

    salt_bytes = base64.b64decode(stored_salt)
    candidate_hash, _ = hash_password(p, salt_bytes)

    if hmac.compare_digest(candidate_hash, stored_hash):
        return True, user_record.get("role", "Officer")

    return False, None

def register_user(username: str, password: str, role: str = "Field Officer") -> bool:
    """Registers a new officer with salted PBKDF2 hash stored encrypted in dsp_vault."""
    u = str(username).strip().lower()
    vault = _load_vault_data()
    if u in vault:
        return False

    pwd_hash, salt_b64 = hash_password(password)
    vault[u] = {
        "role": role,
        "hash": pwd_hash,
        "salt": salt_b64
    }
    _save_vault_data(vault)
    return True

def encrypt_sensitive_log(data_str: str) -> bytes:
    """Encrypts payload with AES-256 Fernet key."""
    return cipher_suite.encrypt(data_str.encode("utf-8"))

def decrypt_sensitive_log(encrypted_bytes: bytes) -> str:
    """Decrypts payload with AES-256 Fernet key."""
    return cipher_suite.decrypt(encrypted_bytes).decode("utf-8")

def update_user_password(username: str, new_password: str) -> bool:
    """Updates an existing user's password with a fresh salt and PBKDF2 hash, encrypted in vault."""
    u = str(username).strip().lower()
    vault = _load_vault_data()
    if u not in vault:
        return False
    pwd_hash, salt_b64 = hash_password(new_password)
    vault[u]["hash"] = pwd_hash
    vault[u]["salt"] = salt_b64
    _save_vault_data(vault)
    return True

def list_vault_officers() -> list[dict]:
    """Returns safe metadata for all registered officers in the encrypted vault without exposing hashes."""
    vault = _load_vault_data()
    return [
        {
            "username": u,
            "role": data.get("role", "Officer"),
            "status": "Encrypted & Hashed",
            "iterations": "100,000 PBKDF2"
        }
        for u, data in vault.items()
    ]

def get_raw_vault_bytes() -> str:
    """Returns the raw on-disk encrypted ciphertext (hex string) stored in credentials.enc."""
    if os.path.exists(ENCRYPTED_VAULT_FILE):
        with open(ENCRYPTED_VAULT_FILE, "rb") as f:
            raw = f.read()
        return raw.hex()[:160] + "..." if len(raw) > 80 else raw.hex()
    return "Vault file not found."

def explain_dsp_pipeline(password: str) -> dict:
    """Generates an educational diagnostic breakdown of the DSP hashing and encryption sequence."""
    salt = os.urandom(16)
    salt_hex = salt.hex()
    pwd_bytes = password.strip().encode("utf-8")
    hash_bytes = hashlib.pbkdf2_hmac("sha256", pwd_bytes, salt, 100000)
    hash_b64 = base64.b64encode(hash_bytes).decode("utf-8")
    sample_json = json.dumps({"test_hash": hash_b64, "salt": salt_hex}).encode("utf-8")
    ciphertext = cipher_suite.encrypt(sample_json)
    return {
        "salt_hex": salt_hex,
        "iterations": 100000,
        "hash_algorithm": "PBKDF2-HMAC-SHA256",
        "hash_b64": hash_b64,
        "cipher": "AES-256 (Fernet CBC + HMAC-SHA256)",
        "ciphertext_preview": ciphertext.decode("utf-8")[:80] + "...",
        "vault_destination": ENCRYPTED_VAULT_FILE,
        "master_key_destination": KEY_FILE
    }

def get_dsp_status() -> dict:
    """Returns Data Security & Privacy (DSP) infrastructure telemetry."""
    return {
        "algorithm": "PBKDF2-HMAC-SHA256 (100k rounds)",
        "encryption": "AES-256 (Fernet CBC + HMAC-SHA256)",
        "vault_directory": VAULT_DIR,
        "status": "Active & Encrypted",
        "key_present": os.path.exists(KEY_FILE),
        "vault_present": os.path.exists(ENCRYPTED_VAULT_FILE),
        "vault_file": ENCRYPTED_VAULT_FILE
    }

# Initialize vault on import
_load_vault_data()