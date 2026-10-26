"""Verify every published file against the preparation manifest."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'MANIFEST.sha256.json').read_text(encoding='utf-8'))
for name,expected in manifest.items():
    path=root/name
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise SystemExit('Hash mismatch: '+name)
print(f'Verified {len(manifest)} published file hashes.')
