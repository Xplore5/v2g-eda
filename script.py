import sys
import secrets
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

NONCE_SIZE = 12 
KEY_SIZE = 32 

def gen_key():
    """Generate a 32-byte random key (hex format)"""
    return secrets.token_hex(KEY_SIZE)

def encrypt_file(infile: Path, outfile: Path, key: bytes):
    """The encrypted file is outfile"""
    data = infile.read_bytes()
    aesgcm = AESGCM(key)
    nonce = secrets.token_bytes(NONCE_SIZE)
    ciphertext = aesgcm.encrypt(nonce, data, None)
    outfile.write_bytes(nonce + ciphertext)

def decrypt_file(infile: Path, outfile: Path, key: bytes):
    """Decrypt the file to outfile"""
    raw = infile.read_bytes()
    nonce = raw[:NONCE_SIZE]
    ciphertext = raw[NONCE_SIZE:]
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    outfile.write_bytes(plaintext)

def encrypt_dir(folder: Path, key: bytes):
    """Recursively encrypt all files in a directory (skipping .enc files)"""
    for p in folder.rglob("*"):
        if p.is_file() and not p.name.endswith(".enc"):
            outp = p.with_suffix(p.suffix + ".enc")
            encrypt_file(p, outp, key)
            print(f"[encryption] {p} -> {outp}")

def decrypt_dir(folder: Path, key: bytes):
    """Recursively decrypt all .enc files in a directory"""
    for p in folder.rglob("*.enc"):
        outp = p.with_suffix("")  # 去掉 .enc
        decrypt_file(p, outp, key)
        print(f"[Decryption] {p} -> {outp}")

def main():
    if len(sys.argv) < 2:
        print("usage:\n"
              "  Generate Key:     python simple_crypto.py gen-key\n"
              "  Encrypted files:     python simple_crypto.py encrypt <infile> <keyhex> [outfile]\n"
              "  Decrypt files:     python simple_crypto.py decrypt <infile> <keyhex> [outfile]\n"
              "  Encrypted Directory:     python simple_crypto.py encrypt-dir <folder> <keyhex>\n"
              "  Decryption Directory:     python simple_crypto.py decrypt-dir <folder> <keyhex>")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "gen-key":
        print(gen_key())
        return

    if cmd in ("encrypt", "decrypt", "encrypt-dir", "decrypt-dir"):
        if len(sys.argv) < 4:
            print("Missing parameters: <path> <keyhex>")
            sys.exit(1)
        path = Path(sys.argv[2])
        try:
            key = bytes.fromhex(sys.argv[3])
            if len(key) != KEY_SIZE:
                raise ValueError
        except Exception:
            print("The key must be a 64-bit hex string (32 bytes)")
            sys.exit(1)

        if cmd == "encrypt":
            outfile = Path(sys.argv[4]) if len(sys.argv) >= 5 else path.with_suffix(path.suffix + ".enc")
            encrypt_file(path, outfile, key)
            print(f"[encryption] {path} -> {outfile}")
        elif cmd == "decrypt":
            outfile = Path(sys.argv[4]) if len(sys.argv) >= 5 else path.with_suffix("")
            decrypt_file(path, outfile, key)
            print(f"[Decryption] {path} -> {outfile}")
        elif cmd == "encrypt-dir":
            encrypt_dir(path, key)
        elif cmd == "decrypt-dir":
            decrypt_dir(path, key)
    else:
        print("Unknown command")

if __name__ == "__main__":
    main()
