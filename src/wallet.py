# wallet.py
import binascii
import os
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.exceptions import InvalidSignature


class Wallet:
    """
    Represents a simple cryptocurrency wallet with key pair management and signing.
    """

    def __init__(self, private_key_file=None, public_key_file=None):
        self.private_key = None
        self.public_key = None
        self.address = None

        if private_key_file and public_key_file and \
           os.path.exists(private_key_file) and os.path.exists(public_key_file):
            self.load_keys(private_key_file, public_key_file)
        else:
            self.generate_key_pair()
            if private_key_file and public_key_file:
                # Save if paths provided
                self.save_keys(private_key_file, public_key_file)

        self._derive_address()

    def generate_key_pair(self):
        """Generates a new ECDSA private/public key pair."""
        # Using SECP256k1 curve, common in cryptocurrencies
        self.private_key = ec.generate_private_key(ec.SECP256K1())
        self.public_key = self.private_key.public_key()
        print("New key pair generated.")

    def _derive_address(self):
        """Derives a simple address from the public key."""
        if not self.public_key:
            return None

        public_bytes = self.public_key.public_bytes(
            # Use DER for a compact, standard representation
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        # Simple address: Hash the DER public key and take hex digest
        # Real addresses often have version bytes, checksums, base58 encoding etc.
        digest = hashes.Hash(hashes.SHA256())
        digest.update(public_bytes)
        self.address = digest.finalize().hex()
        print(f"Wallet Address: {self.address}")

    def get_public_key_pem(self):
        """Returns the public key in pem format (uncompressed)."""
        if not self.public_key:
            return None
        # Use PEM encoding for easy readability/storage, though larger than DER/Compressed
        pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        # Return the core part without header/footer for simplicity in transactions
        # A better way might be specific encoding like DER or Compressed Point
        return pem.decode('utf-8')  # Keep as PEM string for now

    def get_public_key_bytes(self):
        """Returns the public key as bytes (DER format)."""
        if not self.public_key:
            return None
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

    @classmethod
    def public_key_from_pem(cls, pem_string):
        """Loads a public key from its PEM hex string."""
        try:
            public_key = serialization.load_pem_public_key(
                pem_string.encode('utf-8')
            )
            return public_key
        except Exception as e:
            print(f"Error loading public key from PEM: {e}")
            return None

    @classmethod
    def get_address_from_public_key(cls, public_key_pem):
        """ Static method to get address directly from PEM public key string """
        public_key = cls.public_key_from_pem(public_key_pem)
        if not public_key:
            return None
        public_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        digest = hashes.Hash(hashes.SHA256())
        digest.update(public_bytes)
        return digest.finalize().hex()

    def sign(self, data):
        """Signs data using the private key."""
        if not self.private_key:
            raise ValueError("Private key not loaded or generated.")
        if isinstance(data, str):
            data = data.encode('utf-8')

        signature = self.private_key.sign(
            data,
            ec.ECDSA(hashes.SHA256())
        )
        # Return hex representation
        return binascii.hexlify(signature).decode('utf-8')

    @staticmethod
    def verify(public_key_pem, signature_hex, data):
        """Verifies a signature using the public key (PEM format)."""
        public_key = Wallet.public_key_from_pem(public_key_pem)
        if not public_key:
            print("Verification failed: Could not load public key.")
            return False

        if isinstance(data, str):
            data = data.encode('utf-8')

        try:
            signature_bytes = binascii.unhexlify(signature_hex)
            public_key.verify(
                signature_bytes,
                data,
                ec.ECDSA(hashes.SHA256())
            )
            return True
        except InvalidSignature:
            # print("Verification failed: Invalid signature.") # Can be noisy
            return False
        except Exception as e:
            print(f"Verification error: {e}")
            return False

    def save_keys(self, private_key_file, public_key_file):
        """Saves the private and public keys to PEM files."""
        if not self.private_key or not self.public_key:
            print("Keys not generated, cannot save.")
            return

        # Save private key
        pem_private = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            # No password protection for simplicity
            encryption_algorithm=serialization.NoEncryption()
        )
        with open(private_key_file, 'wb') as f:
            f.write(pem_private)
        print(f"Private key saved to {private_key_file}")

        # Save public key
        pem_public = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        with open(public_key_file, 'wb') as f:
            f.write(pem_public)
        print(f"Public key saved to {public_key_file}")

    def load_keys(self, private_key_file, public_key_file):
        """Loads private and public keys from PEM files."""
        try:
            # Load private key
            with open(private_key_file, 'rb') as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None  # No password expected
                )
            # Derive public key from loaded private key
            self.public_key = self.private_key.public_key()
            print(f"Keys loaded from {private_key_file} and {public_key_file}")

            # Verify the loaded public key matches the one in the file (optional sanity check)
            with open(public_key_file, 'rb') as f:
                loaded_public_pem = f.read()
            derived_public_pem = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            if loaded_public_pem != derived_public_pem:
                print(
                    "WARNING: Loaded public key file does not match derived public key.")
                # Decide how to handle: trust derived key, or fail? Trusting derived is safer.

        except FileNotFoundError:
            print(f"Error loading keys: File not found.")
            self.private_key = None
            self.public_key = None
        except Exception as e:
            print(f"Error loading keys: {e}")
            self.private_key = None
            self.public_key = None
