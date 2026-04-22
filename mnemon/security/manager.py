"""
Mnemon Security Layer
Tenant isolation, content classification, sensitive data filtering,
data residency enforcement, and encryption at rest for privileged content.
"""

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# SENSITIVITY CLASSIFICATION
# ─────────────────────────────────────────────

class ContentSensitivity(Enum):
    PUBLIC       = "public"       # no restrictions
    INTERNAL     = "internal"     # tenant-only, standard storage
    CONFIDENTIAL = "confidential" # encrypted at rest
    PRIVILEGED   = "privileged"   # legal/HR/medical — encrypted, audit-logged


# ─────────────────────────────────────────────
# CONTENT FILTER
# ─────────────────────────────────────────────

class ContentFilter:
    """
    Pre-write content filter.
    Categories configured per tenant — matching content passes through
    without being stored. Agent still gets the info for current task.
    Nothing persists.
    """

    BUILTIN_PATTERNS: Dict[str, List[str]] = {
        "pii": [
            r"\b\d{3}-\d{2}-\d{4}\b",           # SSN
            r"\b\d{16}\b",                         # credit card
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # email
            r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", # phone
        ],
        "medical_records": [
            r"\bdiagnos[ie]s\b", r"\bprescription\b", r"\bpatient\s+id\b",
            r"\bmedical\s+record\b", r"\btreatment\s+plan\b",
        ],
        "financial_pii": [
            r"\baccount\s+number\b", r"\brouting\s+number\b",
            r"\btax\s+id\b", r"\bein\b", r"\bssn\b",
        ],
        "legal_privilege": [
            r"\battorney.client\b", r"\bprivileged\s+communication\b",
            r"\bwork\s+product\b", r"\blegal\s+hold\b",
        ],
        "hr_data": [
            r"\bperformance\s+review\b", r"\bsalary\b", r"\bterminat\w+\b",
            r"\bdisciplinary\b", r"\bhr\s+record\b",
        ],
    }

    def __init__(self, blocked_categories: Optional[List[str]] = None):
        self.blocked = set(blocked_categories or [])
        self._patterns: Dict[str, List[re.Pattern]] = {}
        for cat, patterns in self.BUILTIN_PATTERNS.items():
            self._patterns[cat] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def should_store(self, content: Any) -> bool:
        """Returns False if content matches a blocked category."""
        if not self.blocked:
            return True
        text = json.dumps(content) if not isinstance(content, str) else content
        for cat in self.blocked:
            if cat in self._patterns:
                for pattern in self._patterns[cat]:
                    if pattern.search(text):
                        logger.debug(f"Content filtered — matched category: {cat}")
                        return False
        return True

    def classify_sensitivity(self, content: Any) -> ContentSensitivity:
        """Classify content sensitivity level."""
        text = json.dumps(content) if not isinstance(content, str) else content
        text_lower = text.lower()

        if any(pattern.search(text) for pattern in self._patterns.get("legal_privilege", [])):
            return ContentSensitivity.PRIVILEGED

        if any(pattern.search(text) for pattern in self._patterns.get("medical_records", [])):
            return ContentSensitivity.PRIVILEGED

        if any(pattern.search(text) for pattern in self._patterns.get("hr_data", [])):
            return ContentSensitivity.CONFIDENTIAL

        if any(pattern.search(text) for pattern in self._patterns.get("pii", [])):
            return ContentSensitivity.CONFIDENTIAL

        if any(pattern.search(text) for pattern in self._patterns.get("financial_pii", [])):
            return ContentSensitivity.CONFIDENTIAL

        return ContentSensitivity.INTERNAL


# ─────────────────────────────────────────────
# ENCRYPTION — AUTO-UPGRADING
# ─────────────────────────────────────────────
#
# Priority order (automatic, zero config):
#   1. cryptography.Fernet — AES-128-CBC + HMAC-SHA256 (pip install mnemon-ai[full])
#   2. XORFallback         — obfuscation only, NOT secure for production
#
# The moment cryptography is installed, Mnemon upgrades silently.
# A key generated without cryptography cannot be decrypted with it — see migration note.
# ─────────────────────────────────────────────

def _try_load_fernet(tenant_key: Optional[str]):
    """Attempt to build a Fernet cipher. Returns (cipher, key_bytes) or None."""
    try:
        from cryptography.fernet import Fernet
        import base64
        # Derive a 32-byte key from the tenant key string
        raw = (tenant_key or "default_dev_key_change_in_production").encode()
        # Pad/truncate to 32 bytes then base64url-encode for Fernet
        key_32 = (raw * (32 // len(raw) + 1))[:32]
        fernet_key = base64.urlsafe_b64encode(key_32)
        cipher = Fernet(fernet_key)
        return cipher
    except ImportError:
        return None
    except Exception as e:
        logger.warning(f"Fernet init failed: {e}")
        return None


class FernetEncryption:
    """
    AES-128-CBC + HMAC-SHA256 encryption via cryptography.Fernet.
    Production-grade. Install: pip install mnemon-ai[full]
    """
    def __init__(self, cipher):
        self._cipher = cipher

    def encrypt(self, data: str) -> str:
        return self._cipher.encrypt(data.encode()).decode()

    def decrypt(self, data: str) -> str:
        return self._cipher.decrypt(data.encode()).decode()


class XORFallbackEncryption:
    """
    XOR obfuscation fallback.
    NOT cryptographically secure — for development only.
    Install cryptography for production: pip install mnemon-ai[full]
    """
    def __init__(self, key: str):
        self._key = key.encode()

    def encrypt(self, data: str) -> str:
        key_bytes = (self._key * (len(data) // len(self._key) + 1))[:len(data)]
        encrypted = bytes(a ^ b for a, b in zip(data.encode(), key_bytes))
        return encrypted.hex()

    def decrypt(self, data: str) -> str:
        encrypted = bytes.fromhex(data)
        key_bytes = (self._key * (len(encrypted) // len(self._key) + 1))[:len(encrypted)]
        return bytes(a ^ b for a, b in zip(encrypted, key_bytes)).decode()


class SimpleEncryption:
    """
    Public encryption interface.
    Automatically selects the best available backend.

    With cryptography installed:
        → FernetEncryption (AES-128-CBC + HMAC — production grade)
    Without:
        → XORFallbackEncryption (obfuscation only — development only)

    To upgrade: pip install mnemon-ai[full]
    No code changes needed.

    WARNING: Content encrypted with XOR cannot be decrypted with Fernet.
    If upgrading a live system, run a one-time migration to re-encrypt stored data.
    """

    def __init__(self, tenant_key: Optional[str] = None, strict: bool = False):
        fernet = _try_load_fernet(tenant_key)
        if fernet:
            self._backend = FernetEncryption(fernet)
            self.backend_name = "fernet-aes"
            self.is_secure = True
            logger.info("Mnemon encryption: Fernet AES-128 active")
        else:
            if strict:
                raise ImportError(
                    "Mnemon encryption: 'cryptography' package is required when "
                    "encrypt_privileged=True or encrypt_confidential=True. "
                    "Run: pip install mnemon-ai[full]"
                )
            self._backend = XORFallbackEncryption(tenant_key or "default_dev_key_change_in_production")
            self.backend_name = "xor-fallback"
            self.is_secure = False
            logger.warning(
                "Mnemon encryption: using XOR fallback (NOT production-safe). "
                "Run: pip install mnemon-ai[full]"
            )

    def encrypt(self, data: str) -> str:
        return self._backend.encrypt(data)

    def decrypt(self, data: str) -> str:
        try:
            return self._backend.decrypt(data)
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return data


# ─────────────────────────────────────────────
# TENANT SECURITY CONFIG
# ─────────────────────────────────────────────

@dataclass
class TenantSecurityConfig:
    """
    Per-tenant security configuration.
    Enforced at write time — never retroactively.
    """
    tenant_id:           str
    data_region:         str = "default"
    blocked_categories:  List[str] = field(default_factory=list)
    encryption_key:      Optional[str] = None
    encrypt_confidential: bool = False
    encrypt_privileged:  bool = True
    require_audit_log:   bool = True
    verification_levels: List[str] = field(default_factory=lambda: ["high", "critical"])

    # Derived
    _filter:    Optional[ContentFilter] = field(default=None, init=False, repr=False)
    _encryptor: Optional[SimpleEncryption] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._filter    = ContentFilter(self.blocked_categories)
        # strict=True when real encryption is actually required — refuses XOR fallback
        needs_encryption = self.encrypt_privileged or self.encrypt_confidential
        self._encryptor = SimpleEncryption(self.encryption_key, strict=needs_encryption)

    def should_store(self, content: Any) -> bool:
        return self._filter.should_store(content)

    def classify(self, content: Any) -> ContentSensitivity:
        return self._filter.classify_sensitivity(content)

    def maybe_encrypt(self, content: Any, sensitivity: ContentSensitivity) -> Any:
        """Encrypt content if sensitivity warrants it."""
        if not self._encryptor:
            return content
        content_str = json.dumps(content) if not isinstance(content, str) else content
        if sensitivity == ContentSensitivity.PRIVILEGED and self.encrypt_privileged:
            return {"__encrypted__": True, "data": self._encryptor.encrypt(content_str)}
        if sensitivity == ContentSensitivity.CONFIDENTIAL and self.encrypt_confidential:
            return {"__encrypted__": True, "data": self._encryptor.encrypt(content_str)}
        return content

    def maybe_decrypt(self, content: Any) -> Any:
        """Decrypt content if it was encrypted."""
        if isinstance(content, dict) and content.get("__encrypted__"):
            try:
                decrypted = self._encryptor.decrypt(content["data"])
                return json.loads(decrypted)
            except Exception as e:
                logger.error(f"Decryption failed: {e}")
                return content
        return content


# ─────────────────────────────────────────────
# SECURITY MANAGER
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# PROMPT INJECTION SCRUBBER
# ─────────────────────────────────────────────

_INJECTION_PATTERNS = [
    re.compile(r'\n\s*Human\s*:', re.IGNORECASE),
    re.compile(r'\n\s*Assistant\s*:', re.IGNORECASE),
    re.compile(r'<\|im_start\|>.*?<\|im_end\|>', re.DOTALL),
    re.compile(r'<\|im_start\|>'),
    re.compile(r'<\|im_end\|>'),
    re.compile(r'<\|endoftext\|>'),
    re.compile(r'ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions?', re.IGNORECASE),
    re.compile(r'disregard\s+(?:all\s+)?(?:previous|prior)\s+instructions?', re.IGNORECASE),
    re.compile(r'you\s+are\s+now\s+(?:a\s+)?(?:an?\s+)?\w+', re.IGNORECASE),
    re.compile(r'\[INST\]|\[/INST\]'),
    re.compile(r'<<SYS>>|<</SYS>>'),
]


def scrub_injection(content: Any) -> Any:
    """
    Strip known prompt-injection markers from content before storing in memory.
    Prevents adversarial memories from hijacking agent context on recall.
    Applied unconditionally at write time.
    """
    if isinstance(content, str):
        for pat in _INJECTION_PATTERNS:
            content = pat.sub('', content)
        return content
    if isinstance(content, dict):
        return {k: scrub_injection(v) for k, v in content.items()}
    return content


class SecurityManager:
    """
    Central security enforcement point.
    Wraps all memory writes with content filtering,
    sensitivity classification, and optional encryption.
    """

    def __init__(self, config: Optional[TenantSecurityConfig] = None):
        self.config = config

    def check_write(self, content: Any) -> Dict[str, Any]:
        """
        Pre-write security check.
        Returns: {allowed, sensitivity, content_to_store}
        """
        if not self.config:
            return {"allowed": True, "sensitivity": ContentSensitivity.INTERNAL, "content": content}

        # Filter check
        if not self.config.should_store(content):
            return {
                "allowed":     False,
                "sensitivity": ContentSensitivity.PRIVILEGED,
                "content":     None,
                "reason":      "blocked_category_match",
            }

        # Classify
        sensitivity = self.config.classify(content)

        # Encrypt if needed
        stored_content = self.config.maybe_encrypt(content, sensitivity)

        return {
            "allowed":     True,
            "sensitivity": sensitivity,
            "content":     stored_content,
        }

    def check_read(self, content: Any) -> Any:
        """Post-read decryption if needed."""
        if not self.config:
            return content
        return self.config.maybe_decrypt(content)

    def get_stats(self) -> Dict:
        if not self.config:
            return {"configured": False}
        return {
            "tenant_id":          self.config.tenant_id,
            "data_region":        self.config.data_region,
            "blocked_categories": self.config.blocked_categories,
            "encryption_enabled": self.config.encrypt_privileged,
        }
