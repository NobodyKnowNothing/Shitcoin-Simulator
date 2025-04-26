# blockchain.py
import hashlib
import datetime
import os
import json
from wallet import Wallet  # Import Wallet for verification and address derivation

# --- Constants ---
MINING_REWARD = 50
DEFAULT_FEE = 1  # Simple fixed transaction fee
DETERMINISTIC_GENESIS_TIMESTAMP = "2024-01-01T00:00:00+00:00"
# --- Transaction Class ---

class Transaction:
    def __init__(self, sender_public_key_pem, receiver_address, amount, fee=DEFAULT_FEE, timestamp=None, signature=None):
        # Sender identified by public key (PEM format string)
        # None sender_public_key indicates a reward/system transaction
        self.sender_public_key_pem = sender_public_key_pem
        self.sender_address = Wallet.get_address_from_public_key(
            sender_public_key_pem) if sender_public_key_pem else "System"
        self.receiver_address = receiver_address
        self.amount = amount
        self.fee = fee
        self.timestamp = timestamp or datetime.datetime.now(
            datetime.timezone.utc).isoformat()
        self.signature = signature  # Hex string of the signature

    def calculate_hash(self):
        """Calculates the SHA-256 hash of the transaction's core data (used for signing)."""
        tx_data = {
            'sender_public_key_pem': self.sender_public_key_pem,
            'receiver_address': self.receiver_address,
            'amount': self.amount,
            'fee': self.fee,
            'timestamp': self.timestamp
        }
        # Use separators=(',', ':') for compact, deterministic JSON
        tx_string = json.dumps(tx_data, sort_keys=True,
                               separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(tx_string).hexdigest()

    def sign(self, wallet):
        """Signs the transaction hash with the provided wallet's private key."""
        # Ensure the wallet provided owns the public key listed as the sender
        # Note: Wallet.get_public_key_pem() should return the PEM format
        if wallet.get_public_key_pem() != self.sender_public_key_pem:
            raise ValueError(
                "Wallet public key (PEM) does not match transaction sender public key.")
        tx_hash = self.calculate_hash()
        # Assuming wallet.sign returns hex
        self.signature = wallet.sign(tx_hash)

    def is_valid(self):
        """Performs basic validation and signature verification."""
        # Reward transactions (identified by None public key) don't need sender signature
        if self.sender_public_key_pem is None or self.sender_address == "System":
            is_reward = True
            # Validate reward structure: Positive amount, receiver exists, no fee (conventionally), no signature
            if not (self.amount > 0 and self.receiver_address and self.signature is None and self.fee == 0):
                print(f"Invalid reward transaction structure: amount={self.amount}, receiver={self.receiver_address}, sig={self.signature}, fee={self.fee}")
                return False
            # Additional checks for reward transactions if needed (e.g., amount limits)
            return True
        else:  # Regular Transaction
            is_reward = False
            # Basic checks for regular transactions
            if not all([self.sender_public_key_pem, self.receiver_address, self.signature]):
                print(f"Invalid TX: Missing field(s) - sender_pk={self.sender_public_key_pem is not None}, receiver={
                    self.receiver_address is not None}, sig={self.signature is not None}")
                return False
            if self.amount <= 0:  # Amount must be positive
                print(f"Invalid TX: Non-positive amount: {self.amount}")
                return False
            if self.fee < 0:  # Fee cannot be negative (can be 0)
                print(f"Invalid TX: Negative fee: {self.fee}")
                return False

            # Verify signature
            tx_hash = self.calculate_hash()
            if not Wallet.verify(self.sender_public_key_pem, self.signature, tx_hash):
                print(f"Invalid TX: Signature verification failed for hash {
                      tx_hash[:8]}...")
                return False

            # Verify derived address matches sender_address (consistency check)
            # This ensures the address stored wasn't tampered with after creation
            derived_address = Wallet.get_address_from_public_key(
                self.sender_public_key_pem)
            if self.sender_address != derived_address:
                print(f"Invalid TX: Stored sender address {
                      self.sender_address} does not match derived address {derived_address}")
                return False

            return True  # All checks passed for regular transaction

    def to_dict(self):
        """Return dictionary representation for serialization."""
        return {
            'sender_public_key_pem': self.sender_public_key_pem,
            'sender_address': self.sender_address,
            'receiver_address': self.receiver_address,
            'amount': self.amount,
            'fee': self.fee,
            'timestamp': self.timestamp,
            'signature': self.signature
        }

    @classmethod
    def from_dict(cls, data):
        """Create a Transaction object from a dictionary."""
        # Determine if it's potentially a reward transaction based on missing/null sender key
        is_reward = data.get('sender_public_key_pem') is None
        # If it's a reward, fee should default to 0, otherwise use provided fee or default fee
        fee = data.get('fee', 0 if is_reward else DEFAULT_FEE)

        return cls(
            sender_public_key_pem=data.get(
                'sender_public_key_pem'),  # Will be None for reward
            receiver_address=data.get(
                'receiver_address'),  # Use .get for safety
            amount=data.get('amount'),  # Use .get for safety
            fee=fee,
            timestamp=data.get('timestamp'),  # Use .get for safety
            signature=data.get('signature')  # Will be None for reward
        )

    def __str__(self):
        sender_disp = self.sender_address[:8] + \
            "..." if self.sender_address and self.sender_address != "System" else "System"
        receiver_disp = self.receiver_address[:8] + \
            "..." if self.receiver_address else "N/A"
        sig_disp = self.signature[:8] + "..." if self.signature else 'N/A'
        return (f"TX [{self.timestamp}] {sender_disp} -> "
                f"{receiver_disp}: {self.amount} (Fee: {self.fee}) Sig: {sig_disp}")

# --- Block Class ---


class Block:
    def __init__(self, index, transactions, timestamp, previous_hash, nonce=0, current_hash=None):
        self.index = index
        # Ensure transactions are stored as dicts internally
        self.transactions = [tx.to_dict() if isinstance(
            tx, Transaction) else tx for tx in transactions]
        self.timestamp = timestamp
        self.previous_hash = previous_hash
        self.nonce = nonce
        # Calculate hash upon creation if not provided (e.g., when loading)
        self.hash = current_hash or self.calculate_hash()

    def calculate_hash(self):
        """Calculates the SHA-256 hash of the block's essential contents."""
        block_content = {
            'index': self.index,
            # Ensure consistent ordering and format of transactions for hashing
            'transactions': json.dumps(self.transactions, sort_keys=True, separators=(',', ':')),
            'timestamp': self.timestamp,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce
        }
        # Use separators=(',', ':') for compact, deterministic JSON hashing
        block_string = json.dumps(
            block_content, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(block_string).hexdigest()

    def mine_block(self, difficulty):
        """
        Simple Proof-of-Work: Increment nonce until the hash starts with
        the required number of zeros ('difficulty').
        """
        target = '0' * difficulty
        # Recalculate hash in case properties changed before mining, or start fresh
        self.hash = self.calculate_hash()
        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self.calculate_hash()
        # print(f"Block {self.index} mined. Nonce: {self.nonce}, Hash: {self.hash}") # Optional logging

    def to_dict(self):
        """Return dictionary representation for serialization."""
        return {
            'index': self.index,
            'transactions': self.transactions,  # Already dicts
            'timestamp': self.timestamp,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
            'hash': self.hash
        }

    @classmethod
    def from_dict(cls, data):
        """Create a Block object from a dictionary."""
        # Basic validation of required fields might be good here
        required_fields = ['index', 'transactions',
                           'timestamp', 'previous_hash']
        if not all(field in data for field in required_fields):
            raise ValueError(
                f"Block data dictionary missing required fields: {data}")

        return cls(
            index=data['index'],
            # Assumes they are already dicts
            transactions=data['transactions'],
            timestamp=data['timestamp'],
            previous_hash=data['previous_hash'],
            nonce=data.get('nonce', 0),  # Default nonce if missing
            current_hash=data.get('hash')  # Pass existing hash if present
        )

    def __str__(self):
        # Convert transaction dicts back to Transaction objects for nice printing
        tx_strs = []
        for tx_data in self.transactions:
            try:
                tx_obj = Transaction.from_dict(tx_data)
                tx_strs.append(f"    {str(tx_obj)}")
            except Exception as e:
                tx_strs.append(f"    [Error parsing tx: {e} Data: {tx_data}]")

        tx_summary = "\n".join(tx_strs) if tx_strs else "    <No Transactions>"
        return (f"Block {self.index} | Hash: {self.hash[:10]}... | Prev: {self.previous_hash[:10]}... "
                f"| Nonce: {self.nonce} | Txs: {len(self.transactions)}\n{tx_summary}")


# --- Blockchain Class ---
class Blockchain:
    DEFAULT_CHAIN_FILE = 'blockchain_data.json' # Shared default file

    def __init__(self, difficulty=4, chain_file=DEFAULT_CHAIN_FILE, genesis_block=None):
        self.difficulty = difficulty
        self.mining_reward = MINING_REWARD
        # Use the provided chain_file path, defaults to the shared one
        self.chain_file = chain_file
        self.chain = [] # Initialize empty

        loaded_chain = []
        if genesis_block:
             # This path is less relevant for Option B but kept for flexibility
            if isinstance(genesis_block, Block) and genesis_block.index == 0:
                self.chain = [genesis_block]
                print("NODE: Initialized chain with provided Genesis Block.")
            else:
                print("Warning: Provided genesis_block is invalid. Falling back to loading/creating.")
                genesis_block = None

        # If no valid genesis_block provided, attempt to load from the SHARED file
        if not self.chain:
             # Attempt to load from the default shared file path
            loaded_chain = self.load_chain(self.chain_file) # Pass the target file path explicitly

        if loaded_chain:
            # Validate the loaded chain structure before accepting
            if self.is_chain_valid(chain_to_validate=loaded_chain, check_transactions=False):
                print(f"NODE: Successfully loaded and validated chain structure from {self.chain_file}")
                self.chain = loaded_chain
                # Optional: Full transaction validation (can be slow)
                # if not self.is_chain_valid(check_transactions=True): ... handle failure
            else:
                print(f"WARNING: Loaded chain from {self.chain_file} is structurally invalid! Creating deterministic Genesis.")
                # Create the deterministic genesis but DO NOT save it back to the shared file here
                self.chain = [self._create_genesis_block()]
                # *** REMOVED self.save_chain() call here ***
        elif not self.chain: # Loading failed (file not found, empty, corrupt) AND no genesis_block provided
            print(f"NODE: No valid chain loaded from {self.chain_file}. Creating deterministic Genesis Block...")
            # Create the deterministic genesis but DO NOT save it back to the shared file here
            self.chain = [self._create_genesis_block()]
            # *** REMOVED self.save_chain() call here ***

    def _create_genesis_block(self):
        """Creates the very first block (index 0) in the chain deterministically."""
        print("NODE: Creating Deterministic Genesis Block...")
        # Use the fixed timestamp for determinism
        genesis_timestamp = DETERMINISTIC_GENESIS_TIMESTAMP
        genesis_block = Block(
            index=0,
            transactions=[],
            timestamp=genesis_timestamp,
            previous_hash="0"
        )
        # Mine the genesis block
        genesis_block.mine_block(self.difficulty) # Use standard difficulty
        print(f"NODE: Deterministic Genesis Block created. Hash: {genesis_block.hash}")
        return genesis_block

    def load_chain(self, file_path): # Accept file_path argument
        """Loads the blockchain from the specified JSON file."""
        try:
            with open(file_path, 'r') as f: # Use the passed file_path
                content = f.read()
                if not content:
                    print(f"Chain file {file_path} is empty.")
                    return []
                chain_data = json.loads(content)
                loaded_chain = [Block.from_dict(block_data) for block_data in chain_data]
                print(f"Blockchain data loaded from {file_path}. Found {len(loaded_chain)} blocks.")
                return loaded_chain
        except FileNotFoundError:
            print(f"Chain file {file_path} not found.")
            return []
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}. File might be corrupted.")
            return []
        except (TypeError, KeyError, ValueError) as e:
            print(f"Error parsing block data from {file_path}: {e}. Data format might be incorrect.")
            return []
        except Exception as e:
            print(f"An unexpected error occurred loading the chain from {file_path}: {e}")
            return []

    def save_chain(self, file_path=None):
        """Saves the current blockchain state to a JSON file.
           Uses file_path if provided, otherwise defaults to self.chain_file.
        """
        target_path = file_path if file_path is not None else self.chain_file # Determine target path
        try:
            # Ensure the directory exists before trying to write the file
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, 'w') as f: # Use target_path
                chain_data = [block.to_dict() for block in self.chain]
                json.dump(chain_data, f, indent=2)
            # Add a log specific to the file being saved
            print(f"Blockchain state saved to {target_path}") # Can be noisy, enable if needed
        except IOError as e:
            print(f"Error saving blockchain to {target_path}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred saving the chain to {target_path}: {e}")

    def get_latest_block(self):
        """Returns the most recent block OBJECT in the chain."""
        # Assumes self.chain is never empty after initialization
        return self.chain[-1]

    def create_new_block(self, transactions, miner_reward_address):
        """
        Creates and mines a new block with the given transactions (list of Transaction objects).
        Includes the mining reward and transaction fees.
        Returns the newly mined Block object.
        Note: This method does NOT add the block to the chain itself.
        """
        if not miner_reward_address:
            raise ValueError("Miner reward address cannot be empty")

        latest_block = self.get_latest_block()
        previous_hash = latest_block.hash
        index = latest_block.index + 1
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Validate incoming transactions before including them
        valid_transactions_for_block = []
        total_fees = 0
        for tx in transactions:
            if not isinstance(tx, Transaction):
                print(f"Warning: Item in transaction list is not a Transaction object: {
                      tx}. Skipping.")
                continue
            if tx.is_valid():  # Perform basic validation and signature check
                # Optional: Add deeper checks here, like sufficient balance (would need get_balance access)
                valid_transactions_for_block.append(tx)
                if tx.sender_address != "System":  # Don't count fee from potential reward txs passed in
                    total_fees += tx.fee
            else:
                print(
                    f"Warning: Invalid transaction provided, will not be included in block: {tx}")

        # Create the reward transaction for the miner
        reward_tx = Transaction(
            sender_public_key_pem=None,  # None indicates system/reward
            receiver_address=miner_reward_address,
            amount=self.mining_reward + total_fees,
            fee=0  # Reward transactions have no fee
            # Timestamp will be generated by Transaction __init__
        )

        # Combine reward transaction and valid user transactions
        # Reward transaction conventionally comes first
        all_transactions_for_block = [reward_tx] + valid_transactions_for_block

        # Create the new block (transactions will be converted to dicts by Block constructor)
        new_block = Block(
            index=index,
            transactions=all_transactions_for_block,
            timestamp=timestamp,
            previous_hash=previous_hash
        )

        # Perform Proof-of-Work mining
        new_block.mine_block(self.difficulty)

        print(f"NODE: Successfully mined new Block {index} with {
              len(all_transactions_for_block)} transactions (including reward).")
        return new_block

    def add_block(self, block_to_add):
        """
        Adds a pre-validated block object to the end of the current chain.
        Assumes the block has already been validated externally (e.g., PoW, basic structure).
        Performs a final check linking it to the current chain tip.
        Returns True if added successfully, False otherwise.
        """
        if not isinstance(block_to_add, Block):
            print("Error: Attempted to add a non-Block object to the chain.")
            return False

        latest_block = self.get_latest_block()

        # Final verification: Does the new block correctly point to our current chain tip?
        if block_to_add.previous_hash != latest_block.hash:
            print(f"Error: Block {block_to_add.index} (Hash: {block_to_add.hash[:8]}...) has incorrect previous_hash "
                  f"({block_to_add.previous_hash[:8]}...). Expected: {latest_block.hash[:8]}...")
            return False

        # Final verification: Is the index correct?
        if block_to_add.index != latest_block.index + 1:
            print(f"Error: Block {block_to_add.index} has incorrect index. Expected: {
                  latest_block.index + 1}")
            return False

        # Optional: Re-verify PoW just to be absolutely sure? Might be redundant if validated before.
        # target = '0' * self.difficulty
        # if not block_to_add.hash.startswith(target) or block_to_add.hash != block_to_add.calculate_hash():
        #     print(f"Error: Block {block_to_add.index} failed final PoW or hash recalculation check before adding.")
        #     return False

        # Add the block to the chain
        self.chain.append(block_to_add)
        print(f"NODE: Block {
              block_to_add.index} successfully added to the chain.")
        # Consider saving the chain immediately after adding a block
        # self.save_chain()
        return True

    def get_balance(self, address):
        """
        Calculates the balance for a given address by iterating through
        all transactions in the current blockchain.
        """
        balance = 0
        if not address or address == "System":
            return 0  # System address has no balance in this model

        # Iterate over Block objects in the current chain
        for block in self.chain:
            # block.transactions contains dictionaries, convert back for logic
            for tx_data in block.transactions:
                try:
                    # Recreate Transaction object to access its properties easily
                    tx = Transaction.from_dict(tx_data)

                    # Subtract if the address is the sender
                    if tx.sender_address == address:
                        balance -= tx.amount
                        balance -= tx.fee  # Sender pays the fee

                    # Add if the address is the receiver
                    if tx.receiver_address == address:
                        balance += tx.amount

                except Exception as e:
                    # Log error if a transaction in a block is malformed
                    print(f"WARN: Error processing transaction data during balance calculation in Block {
                          block.index}: {e} \n   Data: {tx_data}")
                    continue  # Skip this malformed transaction

        return balance

    def is_chain_valid(self, chain_to_validate=None, check_transactions=True, difficulty=None):
        """
        Validates a given chain (list of Block objects) or the instance's chain.
        Checks: Genesis block, block links (previous_hash), hash recalculation,
        Proof-of-Work difficulty, and optionally transaction validity within each block.
        """
        chain = chain_to_validate if chain_to_validate is not None else self.chain
        effective_difficulty = difficulty if difficulty is not None else self.difficulty
        target = '0' * effective_difficulty

        if not chain:
            print("Validation Error: Chain is empty.")
            return False

        # 1. Validate Genesis Block
        genesis_block = chain[0]
        if not isinstance(genesis_block, Block):
            print("Validation Error: Genesis element is not a Block object.")
            return False
        if genesis_block.index != 0:
            print(f"Validation Error: Genesis block index is not 0 (is {
                  genesis_block.index}).")
            return False
        if genesis_block.previous_hash != "0":
            print(f"Validation Error: Genesis block previous_hash is not '0' (is '{
                  genesis_block.previous_hash}').")
            return False
        # Check genesis hash and PoW
        if genesis_block.hash != genesis_block.calculate_hash():
            print(f"Validation Error: Genesis block hash mismatch. Stored: {
                  genesis_block.hash[:8]}, Calculated: {genesis_block.calculate_hash()[:8]}")
            return False
        # Genesis PoW might use a different difficulty, handle carefully or adjust check
        # Assuming genesis uses the same difficulty for now:
        if not genesis_block.hash.startswith(target):
            print(f"Validation Error: Genesis block PoW insufficient for difficulty {
                  effective_difficulty}. Hash: {genesis_block.hash}")
            # If genesis had diff=1: target_gen = '0' * 1; if not genesis_block.hash.startswith(target_gen): return False
            # For simplicity, assume standard difficulty applies unless specified otherwise
            # return False # Uncomment if enforcing standard difficulty on genesis

        # 2. Validate Subsequent Blocks
        for i in range(1, len(chain)):
            current_block = chain[i]
            previous_block = chain[i-1]  # We already validated chain[0]

            if not isinstance(current_block, Block):
                print(f"Validation Error: Element at index {
                      i} is not a Block object.")
                return False

            # a. Check index sequence
            if current_block.index != previous_block.index + 1:
                print(f"Validation Error: Block {
                      current_block.index} index mismatch. Previous index was {previous_block.index}.")
                return False

            # b. Check hash recalculation
            if current_block.hash != current_block.calculate_hash():
                print(f"Validation Error: Block {current_block.index} hash mismatch. Stored: {
                      current_block.hash[:8]}, Calculated: {current_block.calculate_hash()[:8]}")
                return False

            # c. Check previous_hash link
            if current_block.previous_hash != previous_block.hash:
                print(f"Validation Error: Block {current_block.index} link broken. Prev_hash: {
                      current_block.previous_hash[:8]}, Expected: {previous_block.hash[:8]}")
                return False

            # d. Check Proof-of-Work
            if not current_block.hash.startswith(target):
                print(f"Validation Error: Block {current_block.index} PoW insufficient for difficulty {
                      effective_difficulty}. Hash: {current_block.hash}")
                return False

            # e. (Optional) Validate transactions within the block
            if check_transactions:
                has_reward_tx = False
                total_fees_in_block = 0
                reward_amount_expected = self.mining_reward  # Start with base reward

                if not current_block.transactions and current_block.index > 0:  # Allow empty genesis
                    print(
                        f"Validation Error: Non-genesis Block {current_block.index} has no transactions.")
                    return False  # Blocks must at least have a reward tx

                for tx_index, tx_data in enumerate(current_block.transactions):
                    try:
                        tx = Transaction.from_dict(tx_data)
                        if not tx.is_valid():  # Use Transaction's own validation
                            print(f"Validation Error: Block {
                                  current_block.index} contains invalid transaction at index {tx_index}: {tx}")
                            return False

                        # Check reward transaction specific rules
                        if tx.sender_address == "System":  # This identifies the reward transaction
                            if tx_index != 0:
                                print(f"Validation Error: Block {
                                      current_block.index} reward transaction is not the first transaction (index {tx_index}).")
                                return False
                            if has_reward_tx:
                                print(f"Validation Error: Block {
                                      current_block.index} has multiple reward transactions.")
                                return False
                            has_reward_tx = True
                            # Store the amount for later comparison
                            reward_amount_found = tx.amount
                        else:
                            # It's a regular transaction, add its fee to the total expected reward
                            total_fees_in_block += tx.fee
                            # Optional deeper validation: Check if sender had sufficient funds *at this point in the chain*
                            # This requires iterating up to the previous block to calculate balance, can be complex/slow.

                    except Exception as e:
                        print(f"Validation Error: Error validating transaction in block {
                              current_block.index} at index {tx_index}: {e}\n   Data: {tx_data}")
                        return False

                # After checking all transactions in the block:
                if current_block.index > 0:  # Genesis doesn't need a reward tx by this convention
                    if not has_reward_tx:
                        print(f"Validation Error: Non-genesis Block {
                              current_block.index} is missing a reward transaction.")
                        return False

                    # Verify the reward amount matches expected reward + collected fees
                    reward_amount_expected += total_fees_in_block
                    if reward_amount_found != reward_amount_expected:
                        print(f"Validation Error: Block {current_block.index} incorrect reward amount. Found: {reward_amount_found}, Expected: {
                              reward_amount_expected} (Reward: {self.mining_reward} + Fees: {total_fees_in_block})")
                        return False

        # If all blocks and checks passed
        return True

    def replace_chain(self, new_chain_objects):
        """
        Replaces the node's current chain with the new one if it's longer and valid.
        Expects a list of Block objects. Returns True if replaced, False otherwise.
        """
        if not isinstance(new_chain_objects, list):
            print("NODE: replace_chain received non-list input.")
            return False
        # Ensure elements are actually Block objects (basic check)
        if new_chain_objects and not all(isinstance(b, Block) for b in new_chain_objects):
            print("NODE: replace_chain received list containing non-Block objects.")
            return False

        current_length = len(self.chain)
        new_length = len(new_chain_objects)

        if new_length > current_length:
            print(f"NODE: Received potentially longer chain (Length: {
                  new_length} vs Current: {current_length}). Validating...")
            # Validate the ENTIRE new chain thoroughly using its own rules (difficulty, transactions)
            if self.is_chain_valid(chain_to_validate=new_chain_objects, check_transactions=True, difficulty=self.difficulty):
                print("NODE: New chain is valid and longer. Replacing current chain.")
                self.chain = new_chain_objects
                self.save_chain()  # Persist the new valid chain
                return True
            else:
                # The incoming longer chain failed validation
                print(
                    "NODE: Received longer chain failed validation. Keeping current chain.")
                return False
        else:
            # Incoming chain is not longer, no action needed
            # print(f"NODE: Received chain (Length: {new_length}) is not longer than current chain (Length: {current_length}). No replacement.") # Can be noisy
            return False

    def __len__(self):
        """Return the number of blocks in the chain."""
        return len(self.chain)

    def __str__(self):
        """Return a string representation of the entire blockchain."""
        chain_str = f"Blockchain (Difficulty: {self.difficulty}, Length: {
            len(self.chain)}): \n"
        chain_str += "=" * 60 + "\n"
        for block_obj in self.chain:
            chain_str += f"{block_obj}\n"  # Uses Block's __str__
            chain_str += "-" * 60 + "\n"
        return chain_str