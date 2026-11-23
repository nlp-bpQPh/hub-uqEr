"""Optional recovery of the exact original data; normal runs use bundled data."""
import hashlib
from pathlib import Path
import urllib.request

URL='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
SHA256='86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
if __name__=='__main__':
    raw=urllib.request.urlopen(URL,timeout=60).read()
    if hashlib.sha256(raw).hexdigest()!=SHA256:
        raise RuntimeError('Source changed: refusing to replace the pinned dataset')
    path=Path(__file__).resolve().parent/'data/input.txt'
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(raw)
    print(f'Verified SHA-256; saved {len(raw)} bytes to {path}')
