# node.py
import threading
import time
import json
import random
import os
import copy
# import traceback # Keep handy for debugging mining errors

from wallet import Wallet
# Make sure Blockchain class is imported correctly
# Assume blockchain.py defines Blockchain, Transaction, Block, DEFAULT_FEE, DETERMINISTIC_GENESIS_TIMESTAMP
from blockchain import Blockchain, Transaction, Block, DEFAULT_FEE, DETERMINISTIC_GENESIS_TIMESTAMP

# --- Global Simulation Settings ---
NODE_REGISTRY = {}
NODE_REGISTRY_LOCK = threading.Lock()
BLOCKCHAIN_DIR = "blockchain_data"
WALLET_DIR = 'wallets'
DIFFICULTY = 4  # This should match the Blockchain class's default or be passed explicitly

# --- Helper Functions ---


def ensure_dir(directory):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# --- P2P Node Class ---


class P2PNode:
    """Represents a node in the simulated P2P network."""

    def __init__(self, node_id, host='localhost', port=None):
        """Initializes a P2P Node."""
        self.node_id = node_id
        self.host = host
        self.port = port  # Port is not actually used in this simulation model
        ensure_dir(WALLET_DIR)
        ensure_dir(BLOCKCHAIN_DIR)
        self.wallet = self._init_wallet()

        # --- Blockchain Initialization ---
        # Instantiate Blockchain. It will attempt to load the shared 'blockchain_data.json'
        # or create the deterministic Genesis block in memory if loading fails.
        self.blockchain = Blockchain(difficulty=DIFFICULTY)

        # --- Node-Specific Chain File Path ---
        # Store the path for this node's specific chain state file
        self._node_chain_file = f"{BLOCKCHAIN_DIR}/{self.node_id}_chain.json"

        # --- Load Node-Specific State (Attempt Recovery) ---
        # Try to load the node's last known state from its specific file.
        # This allows resuming from a previous state if the file exists and is valid/longer.
        self.load_node_specific_chain()  # Attempt to load persisted state

        # --- Node State Variables ---
        self.pending_transactions = {}  # {tx_hash: Transaction object}
        self.known_tx_hashes = set()   # Hashes of txs in chain + pending pool
        # {node_id: P2PNode object} - Direct references in simulation
        self.peers = {}
        # Protects access to shared node resources (chain, pending, peers)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()  # Signals threads to stop gracefully
        self.mining_thread = threading.Thread(
            target=self._mine_loop, daemon=True)
        # Queue for incoming messages (tx, block)
        self.message_queue = []
        self.processing_thread = threading.Thread(
            target=self._process_queue, daemon=True)

        # --- Orphan Block Handling ---
        # Blocks received whose parents are not yet known
        self.orphan_blocks = {}        # {block_hash: Block object}
        # {needed_parent_hash: set(orphan_block_hashes)}
        self.orphan_parent_needed = {}

        # Populate known_tx_hashes from the current state of the blockchain
        self._update_known_tx_hashes()

    def _init_wallet(self):
        """Initializes or loads the node's wallet."""
        priv_file = f"{WALLET_DIR}/{self.node_id}_priv.pem"
        pub_file = f"{WALLET_DIR}/{self.node_id}_pub.pem"
        print(f"NODE {self.node_id}: Initializing wallet...")
        # Wallet() handles creating keys if files don't exist, or loading them if they do.
        return Wallet(private_key_file=priv_file, public_key_file=pub_file)

    def _get_node_chain_file(self):
        """Gets the path for this node's specific chain state file."""
        return self._node_chain_file

    def save_blockchain(self):
        """Saves the node's current blockchain state TO ITS NODE-SPECIFIC FILE."""
        node_specific_file = self._get_node_chain_file()
        print(f"NODE {self.node_id}: Attempting to save chain state to {
              node_specific_file}")
        # Uses the Blockchain class's save method, passing the node-specific path
        self.blockchain.save_chain(file_path=node_specific_file)
        print(f"NODE {self.node_id}: Blockchain state saved to {
              node_specific_file}.")

    def load_node_specific_chain(self):
        """
        Attempts to load the blockchain state from this node's specific file.
        If successful and the loaded chain is valid and longer than the
        one loaded/created during initial Blockchain instantiation, it replaces it.
        """
        node_specific_file = self._get_node_chain_file()
        if os.path.exists(node_specific_file):
            print(f"NODE {self.node_id}: Found node-specific chain file {
                  node_specific_file}. Attempting load...")
            # Use a temporary Blockchain instance to load the data
            loader = Blockchain(difficulty=self.blockchain.difficulty)
            loaded_specific_chain_objects = loader.load_chain(
                node_specific_file)  # Returns list of Block objects or None

            if loaded_specific_chain_objects:
                # Validate the loaded chain (basic structure, hashes, PoW)
                # Use check_transactions=False for faster loading check initially.
                # Full validation happens during runtime checks anyway.
                if loader.is_chain_valid(chain_to_validate=loaded_specific_chain_objects, check_transactions=False):
                    # Compare with the initially loaded/created chain in self.blockchain
                    with self.lock:
                        # Make sure the initial chain exists (should always have Genesis)
                        if not self.blockchain.chain:
                            print(f"NODE {
                                  self.node_id}: ERROR - Initial blockchain is empty during specific load check.")
                            return

                        if len(loaded_specific_chain_objects) > len(self.blockchain.chain):
                            # Ensure it forks correctly from the same genesis block
                            if loaded_specific_chain_objects[0].hash == self.blockchain.chain[0].hash:
                                print(f"NODE {
                                      self.node_id}: Node-specific chain is longer and valid. Overriding initial chain.")
                                # Replace the chain object list directly
                                self.blockchain.chain = loaded_specific_chain_objects
                                # Note: Transaction pool reconciliation isn't done here, relies on _update_known_tx_hashes
                                # and subsequent block processing. This is a simplification.
                            else:
                                print(f"NODE {self.node_id}: Node-specific chain has different genesis ({
                                      loaded_specific_chain_objects[0].hash[:8]} vs {self.blockchain.chain[0].hash[:8]}). Ignoring.")
                        else:
                            print(f"NODE {
                                  self.node_id}: Node-specific chain is not longer than initial chain. Ignoring.")
                else:
                    print(f"NODE {
                          self.node_id}: Node-specific chain file {node_specific_file} is invalid. Ignoring.")
            else:
                print(f"NODE {self.node_id}: Failed to load data from node-specific chain file {
                      node_specific_file}. Ignoring.")
        # else: Node-specific file doesn't exist, node starts with initial chain.

    def _update_known_tx_hashes(self):
        """Populates the set of known transaction hashes from the current blockchain."""
        # Should be called whenever the chain state changes significantly (init, block add, chain replace)
        with self.lock:
            self.known_tx_hashes.clear()
            for block in self.blockchain.chain:
                for tx_data in block.transactions:
                    # We need to reconstruct the transaction hash consistently
                    try:
                        # Recreate object to calculate hash as it would be when received/created
                        tx_obj_temp = Transaction.from_dict(tx_data)
                        tx_hash = tx_obj_temp.calculate_hash()  # Hash used for identification
                        self.known_tx_hashes.add(tx_hash)
                    except Exception as e:
                        # Log error if a transaction in a block fails to parse/hash
                        print(f"NODE {self.node_id}: WARNING - Error processing tx data in block {
                              block.index} during hash update: {e}")
                        continue  # Skip malformed tx data if any

            # Also ensure pending transactions are considered known
            for tx_hash in self.pending_transactions:
                self.known_tx_hashes.add(tx_hash)

    def connect_to_peers(self):
        """Simulates discovering and 'connecting' to other nodes via the global registry."""
        with NODE_REGISTRY_LOCK:
            # Connect to all other registered nodes
            self.peers = {nid: node for nid,
                          node in NODE_REGISTRY.items() if nid != self.node_id}
        print(f"NODE {self.node_id}: Connected to peers: {
              list(self.peers.keys())}")

    def start(self):
        """Starts the node's background threads (mining and message processing)."""
        print(f"NODE {self.node_id}: Starting...")
        # Find peers (must happen after all nodes are created and registered)
        self.connect_to_peers()
        self.mining_thread.start()
        self.processing_thread.start()
        print(f"NODE {self.node_id}: Mining and processing threads started.")

    def stop(self):
        """Signals the node's threads to stop and waits for them to join."""
        print(f"NODE {self.node_id}: Stopping...")
        self.stop_event.set()
        # Add a dummy message to ensure the processing thread wakes up if idle
        # Enqueueing None will be handled gracefully
        self.enqueue_message(None)
        try:
            self.mining_thread.join(timeout=2)
            self.processing_thread.join(timeout=2)
        except Exception as e:
            print(f"NODE {self.node_id}: Error joining threads: {e}")
        self.save_blockchain()  # Save the final state
        print(f"NODE {self.node_id}: Stopped.")

    # --- Communication Methods (Simulated Network via Queues) ---

    def broadcast_transaction(self, transaction):
        """Sends a transaction object to all known peers by adding it to their queues."""
        tx_hash = transaction.calculate_hash()
        print(f"NODE {self.node_id}: Broadcasting Tx {tx_hash[:8]}...")
        tx_data = transaction.to_dict()  # Serialize for 'sending'
        with self.lock:
            # Copy peers to avoid holding lock while calling other nodes' methods
            peers_copy = dict(self.peers)

        for peer_id, peer_node in peers_copy.items():
            try:
                # Simulate sending over network by enqueueing the message tuple
                peer_node.enqueue_message(('transaction', tx_data))
            except Exception as e:
                # Catch potential errors if a peer node is malfunctioning or stopped
                print(f"NODE {self.node_id}: Error sending tx to peer {
                      peer_id}: {e}")

    def broadcast_block(self, block):
        """Sends a block object to all known peers by adding it to their queues."""
        print(f"NODE {self.node_id}: Broadcasting Block {
              block.index} [{block.hash[:8]}]...")
        block_data = block.to_dict()  # Serialize for 'sending'
        with self.lock:
            # Copy peers to avoid holding lock while calling other nodes' methods
            peers_copy = dict(self.peers)

        for peer_id, peer_node in peers_copy.items():
            try:
                # Simulate sending over network
                peer_node.enqueue_message(('block', block_data))
            except Exception as e:
                print(f"NODE {self.node_id}: Error sending block to peer {
                      peer_id}: {e}")

    def enqueue_message(self, message):
        """Adds an incoming message (e.g., from broadcast) to the node's processing queue."""
        # In a real system, this would be called by the network listening component
        self.message_queue.append(message)

    # --- Processing Logic ---

    def _process_queue(self):
        """Target method for the processing thread. Handles messages from the queue."""
        print(f"NODE {self.node_id}: Message processing thread started.")
        while not self.stop_event.is_set():
            message = None
            try:
                if self.message_queue:
                    message = self.message_queue.pop(0)  # FIFO processing

                if message:
                    msg_type, msg_data = message
                    if msg_type == 'transaction':
                        self.handle_received_transaction(msg_data)
                    elif msg_type == 'block':
                        self.handle_received_block(msg_data)
                    # Add handlers for other message types (e.g., block requests) if needed
                else:
                    # If queue is empty, wait briefly to avoid busy-waiting
                    # Use the event's wait method for efficient sleeping
                    self.stop_event.wait(timeout=0.1)

            except Exception as e:
                print(f"NODE {self.node_id}: ERROR in processing queue: {e}")
                # Optionally add traceback here for debugging queue errors
                import traceback
                traceback.print_exc()
                time.sleep(0.5)  # Prevent rapid error loops

            # Check stop event again after processing/sleeping
            # Exit loop if stop is signaled AND the queue is empty to process final messages
            if self.stop_event.is_set() and not self.message_queue:
                break
        print(f"NODE {self.node_id}: Message processing thread finished.")

    def handle_received_transaction(self, tx_data):
        """Processes a transaction received from a peer."""
        try:
            # Deserialize transaction data
            transaction = Transaction.from_dict(tx_data)
            if not transaction:  # Ensure deserialization worked
                print(
                    f"NODE {self.node_id}: Received malformed Tx data. Discarding.")
                return

            tx_hash = transaction.calculate_hash()

            with self.lock:  # Acquire lock to check node state
                # 1. Check if already known (in chain or pending pool)
                if tx_hash in self.known_tx_hashes:
                    # This is common and expected, uncomment for verbose logging
                    # print(f"NODE {self.node_id}: Received known Tx {tx_hash[:8]}. Ignoring.")
                    return

                # 2. Basic validation (e.g., signature, structure - handled by is_valid)
                # Pass the current blockchain state (needed for UTXO check in real scenario)
                # For balance model, we check balance separately below.
                if not transaction.is_valid():  # Assumes is_valid checks signature etc.
                    print(f"NODE {self.node_id}: Received invalid Tx {
                          tx_hash[:8]} (failed is_valid check). Discarding.")
                    return

                # 3. Check sender balance (using THIS node's current blockchain view)
                # Reward transactions ("System" sender) don't need this check
                if transaction.sender_address != "System":
                    sender_balance = self.blockchain.get_balance(
                        transaction.sender_address)
                    required_amount = transaction.amount + transaction.fee
                    if sender_balance < required_amount:
                        print(f"NODE {self.node_id}: Received Tx {tx_hash[:8]} invalid (Insufficient funds: {
                              sender_balance} < {required_amount}). Discarding.")
                        # Note: This tx might be valid on another node's fork.
                        return

                # 4. If valid and new, add to local pending pool and known set
                self.pending_transactions[tx_hash] = transaction
                self.known_tx_hashes.add(tx_hash)
                print(f"NODE {self.node_id}: Added received Tx {
                      tx_hash[:8]} to pending pool ({len(self.pending_transactions)} total).")

            # 5. Re-broadcast to peers (Simple Flooding Gossip)
            # Only rebroadcast transactions that were new and valid for this node
            self.broadcast_transaction(transaction)

        except Exception as e:
            print(f"NODE {self.node_id}: ERROR processing received transaction: {
                  e} data={tx_data}")
            # Optionally add traceback
            # import traceback
            # traceback.print_exc()

    def handle_received_block(self, block_data):
        """Processes a block received from a peer, handling forks, orphans, and chain updates."""
        try:
            # Deserialize block data
            received_block = Block.from_dict(block_data)
            if not received_block:  # Ensure deserialization worked
                print(
                    f"NODE {self.node_id}: Received malformed block data. Discarding.")
                return

            received_hash = received_block.hash  # Use the hash stored in the block object
            received_prev_hash = received_block.previous_hash

            # --- Basic Validation Checks ---
            # 1. Recalculate hash to ensure data integrity and stored hash matches
            calculated_hash = received_block.calculate_hash()
            if received_hash != calculated_hash:
                print(f"NODE {self.node_id}: Received block {received_block.index} has INCORRECT stored hash ({
                      received_hash[:8]} vs calculated {calculated_hash[:8]}). Discarding.")
                return

            # 2. Proof-of-Work check
            target = '0' * self.blockchain.difficulty
            if not received_hash.startswith(target):
                print(f"NODE {self.node_id}: Received block {received_block.index} [{
                      received_hash[:8]}] failed PoW check (difficulty {self.blockchain.difficulty}). Discarding.")
                return

            # 3. Basic Timestamp check (optional, can be complex)
            # e.g., received_block.timestamp <= time.time() + ALLOWED_FUTURE_TIME_SECONDS
            # e.g., received_block.timestamp > latest_block.timestamp

            with self.lock:  # Acquire lock for accessing/modifying chain state
                # --- Quick Redundancy Checks ---
                # Check if block is already in the main chain
                if any(b.hash == received_hash for b in self.blockchain.chain):
                    # print(f"NODE {self.node_id}: Received block {received_hash[:8]} already in main chain. Ignoring.")
                    return
                # Check if block is already stored as an orphan
                if received_hash in self.orphan_blocks:
                    # print(f"NODE {self.node_id}: Received block {received_hash[:8]} already known as an orphan. Ignoring.")
                    return

                # --- Identify Current State ---
                latest_block = self.blockchain.get_latest_block()

                # --- Processing Cases ---

                # Case 1: Block extends the current main chain tip (Common case)
                if received_prev_hash == latest_block.hash and received_block.index == latest_block.index + 1:
                    print(f"NODE {self.node_id}: Received block {
                          received_block.index} [{received_hash[:8]}] extends current tip.")
                    # Validate the block's contents (transactions, etc.) in the context of the current chain.
                    # Blockchain.add_block should perform these checks.
                    # add_block validates before adding
                    if self.blockchain.add_block(received_block):
                        print(f"NODE {self.node_id}: New block {
                              received_block.index} is valid. Added to chain.")
                        # Remove transactions included in the block from the pending pool
                        removed_count = self._remove_txs_from_pending(
                            received_block)
                        print(f"NODE {self.node_id}: Removed {
                              removed_count} confirmed txs from pool.")
                        # Persist the updated chain state
                        self.save_blockchain()
                        # Check if this newly added block unlocks any orphans
                        self._process_orphans(received_hash)
                        # Broadcast the valid block we just added (outside lock if possible?)
                        # For simplicity, keep broadcast call here, but release lock before network IO in real system.
                        # Block object needs to be passed to broadcast
                        block_to_broadcast = received_block  # Use the object we just added
                    else:
                        # add_block failed validation (e.g., invalid tx, bad timestamp, etc.)
                        print(f"NODE {self.node_id}: Received block {
                              received_block.index} failed validation via add_block. Discarding.")
                        # No state change, no broadcast, no orphan check needed.
                        return  # Exit processing for this block

                    # --- Broadcast needs to happen *after* releasing the lock ideally ---
                    # We'll broadcast after the lock block for simulation simplicity
                    # self.broadcast_block(block_to_broadcast) # Moved outside lock below

                # Case 2: Block's parent is in the main chain, but not the tip (Potential fork)
                elif any(b.hash == received_prev_hash for b in self.blockchain.chain):
                    print(f"NODE {self.node_id}: Received block {received_block.index} [{
                          received_hash[:8]}] connects to known ancestor {received_prev_hash[:8]}. Potential fork.")
                    # This block represents a potential alternative chain.
                    # Check if this block implies a *longer* chain than our current one.
                    if received_block.index > latest_block.index:
                        print(f"NODE {self.node_id}: Block {received_block.index} is further ahead than current tip {
                              latest_block.index}. Possible longer chain detected.")
                        # In a real network, we'd request missing blocks between the ancestor and this block.
                        # In this simulation, we trigger a general chain sync request to evaluate all peers.
                        # Use a flag or direct call, but do it AFTER releasing the lock.
                        needs_sync_check = True
                        # Store block temporarily in orphans until sync clarifies? Or just trigger sync?
                        # Let's just trigger sync for simplicity.
                        # self.request_chain_from_peers() # Call this after lock release

                    else:
                        # It connects to an older block but doesn't make the chain longer (shorter fork).
                        print(f"NODE {self.node_id}: Block {received_block.index} connects to ancestor but doesn't represent longer chain ({
                              received_block.index} <= {latest_block.index}). Ignoring (shorter fork).")
                        # Do nothing, stick with the current main chain.
                    # Finished processing this case (either sync triggered or ignored)
                    return

                # Case 3: Block's parent is NOT known in the main chain (Orphan)
                else:
                    print(f"NODE {self.node_id}: Received ORPHAN block {received_block.index} [{
                          received_hash[:8]}]. Parent {received_prev_hash[:8]} unknown. Storing.")
                    # Store the orphan block object itself
                    self.orphan_blocks[received_hash] = received_block
                    # Record which parent hash this orphan needs
                    if received_prev_hash not in self.orphan_parent_needed:
                        self.orphan_parent_needed[received_prev_hash] = set()
                    self.orphan_parent_needed[received_prev_hash].add(
                        received_hash)
                    print(f"NODE {self.node_id}: Orphan count: {
                          len(self.orphan_blocks)}")

                    # Optional: Request the missing parent block from peers
                    # self.request_block_from_peers(received_prev_hash) # Needs implementation if desired

                    return  # Finished processing this case (stored as orphan)

            # --- Actions after releasing lock (if needed) ---
            if 'block_to_broadcast' in locals():
                # If Case 1 happened and block was added successfully
                self.broadcast_block(block_to_broadcast)

            if 'needs_sync_check' in locals() and needs_sync_check:
                # If Case 2 happened and suggested a longer chain
                print(f"NODE {
                      self.node_id}: Triggering chain sync check due to potentially longer fork block received.")
                self.request_chain_from_peers()  # Initiate sync check

        except Exception as e:
            print(f"NODE {self.node_id}: FATAL ERROR processing received block: {
                  e} data={block_data}")
            import traceback
            traceback.print_exc()  # Print stack trace for debugging block processing errors

    def _process_orphans(self, newly_added_block_hash):
        """
        Checks if a newly added block is the parent needed by any stored orphans.
        If so, it attempts to process those orphans recursively.
        Assumes the caller holds the necessary lock.
        """
        # Check if the hash of the block just added is listed as a needed parent
        if newly_added_block_hash in self.orphan_parent_needed:
            # Get the set of orphan block hashes that were waiting for this parent
            orphans_to_process_hashes = self.orphan_parent_needed.pop(
                newly_added_block_hash)
            print(f"NODE {self.node_id}: Block {newly_added_block_hash[:8]} unlocks orphans: {
                  [h[:8] for h in orphans_to_process_hashes]}")

            # Retrieve the actual Block objects from the orphan pool
            orphans_to_process_blocks = []
            # Prevent potential infinite loops in edge cases
            processed_hashes_this_round = set()
            for orphan_hash in list(orphans_to_process_hashes):  # Iterate copy
                if orphan_hash in self.orphan_blocks:
                    orphans_to_process_blocks.append(
                        self.orphan_blocks.pop(orphan_hash))
                    processed_hashes_this_round.add(orphan_hash)
                else:
                    # This might happen if the orphan was received and processed through another path concurrently
                    print(f"WARN {self.node_id}: Orphan hash {
                          orphan_hash[:8]} expected but not found in orphan_blocks during processing.")

            # Sort by block index to process them in the correct order (important!)
            orphans_to_process_blocks.sort(key=lambda b: b.index)

            # Process these now-potentially-connectable blocks
            # Use handle_received_block which contains all necessary checks and logic
            for orphan_block in orphans_to_process_blocks:
                print(f"NODE {self.node_id}: Attempting to process unlocked orphan {
                      orphan_block.index} [{orphan_block.hash[:8]}]...")
                # Re-submit the block data to the handler. It should now hopefully match Case 1 or Case 2.
                # Pass the block's dictionary representation.
                # Need to release and re-acquire lock? handle_received_block acquires lock.
                # This recursive call within the lock could be problematic.
                # Safer approach: Add to a temporary list and process after releasing the main lock, or re-queue.
                # For simulation simplicity, direct call (be wary of deep recursion / lock contention)
                # ***** Re-consider recursive lock acquisition *****
                # Let's try re-queuing instead for safety.
                # self.handle_received_block(orphan_block.to_dict()) # <-- Potentially risky recursive lock
                # --- Safer: Re-queue the orphan block ---
                print(
                    f"NODE {self.node_id}: Re-queueing orphan {orphan_block.index} for processing.")
                # Use a special marker? No, just enqueue normally.
                self.enqueue_message(('block', orphan_block.to_dict()))

    def request_chain_from_peers(self):
        """
        Simulates requesting the blockchain from all connected peers.
        Compares received chains and potentially replaces the current chain
        with the longest valid one found. (SIMULATION ONLY)
        """
        print(
            f"NODE {self.node_id}: Initiating chain sync request with peers...")
        with self.lock:
            # Copy peers list to avoid holding lock during potentially long peer interactions
            peers_copy = dict(self.peers)
            # Get a copy of the current chain to avoid modification issues during comparison
            # And get current length/tip under lock
            current_chain_copy = copy.deepcopy(self.blockchain.chain)
            current_length = len(current_chain_copy)
            current_tip_hash = current_chain_copy[-1].hash if current_chain_copy else None

        # Information about the best chain found so far
        best_chain_info = {
            'length': current_length,
            'chain_obj_list': current_chain_copy,  # Store Block objects
            'source_peer_id': self.node_id  # Initially, self is best
        }

        print(f"NODE {self.node_id}: Current chain length = {current_length}, Tip = {
              current_tip_hash[:8] if current_tip_hash else 'N/A'}")

        # Iterate through peers and request their chain state
        for peer_id, peer_node in peers_copy.items():
            print(f"NODE {self.node_id}: Requesting chain from peer {
                  peer_id}...")
            try:
                # In simulation, directly call a method on the peer to get its chain
                # This method MUST return a deep copy to prevent shared state issues
                peer_chain_objects = peer_node.get_chain_copy()  # Assumes peer has this method

                if not peer_chain_objects:
                    print(f"NODE {self.node_id}: Received empty chain from {
                          peer_id}. Skipping.")
                    continue

                peer_chain_len = len(peer_chain_objects)
                peer_tip_hash = peer_chain_objects[-1].hash if peer_chain_objects else None
                print(f"NODE {self.node_id}: Received chain from {peer_id} (Length: {
                      peer_chain_len}, Tip: {peer_tip_hash[:8] if peer_tip_hash else 'N/A'})")

                # --- Comparison Logic ---
                # 1. Check if the peer's chain is potentially better (longer)
                if peer_chain_len > best_chain_info['length']:
                    print(f"NODE {self.node_id}: Chain from {peer_id} is longer ({
                          peer_chain_len} > {best_chain_info['length']}). Validating...")

                    # 2. Validate the received chain *thoroughly*
                    # Use a temporary Blockchain instance for validation context if needed,
                    # or rely on the static/instance methods of the existing Blockchain class.
                    # Use the current node's difficulty setting.
                    # Temp instance for validation context
                    validator = Blockchain(
                        difficulty=self.blockchain.difficulty)
                    # Ensure the genesis block matches! Crucial check.
                    if not self.blockchain.chain or peer_chain_objects[0].hash != self.blockchain.chain[0].hash:
                        print(f"NODE {self.node_id}: Chain from {
                              peer_id} has different Genesis block! Ignoring.")
                        continue

                    if validator.is_chain_valid(chain_to_validate=peer_chain_objects, check_transactions=True):
                        print(f"NODE {self.node_id}: Valid longer chain found from {
                              peer_id} (Length: {peer_chain_len}). Updating best found.")
                        # Update the best chain found so far
                        best_chain_info['length'] = peer_chain_len
                        # Store the actual Block objects
                        best_chain_info['chain_obj_list'] = peer_chain_objects
                        best_chain_info['source_peer_id'] = peer_id
                    else:
                        print(f"NODE {self.node_id}: Chain from {
                              peer_id} is longer but INVALID. Ignoring.")
                # Optional: Handle equal length chains (e.g., prefer based on tip hash, or random) - Skipped for simplicity

            except Exception as e:
                print(f"NODE {
                      self.node_id}: Error requesting/processing chain from peer {peer_id}: {e}")
                # import traceback # Uncomment for debugging peer communication issues
                # traceback.print_exc()

        # --- After checking all peers ---
        # Acquire lock again to perform the chain replacement if needed
        with self.lock:
            # Check if a better chain was found from a peer
            if best_chain_info['source_peer_id'] != self.node_id:
                print(f"NODE {self.node_id}: Best chain found from {best_chain_info['source_peer_id']} (Length: {
                      best_chain_info['length']}). Attempting replacement.")
                # Use the Blockchain's internal replace_chain method which should handle validation again (defense-in-depth)
                # Pass the list of Block objects
                if self.blockchain.replace_chain(best_chain_info['chain_obj_list']):
                    print(f"NODE {self.node_id}: Chain successfully replaced.")
                    # CRITICAL: Update node's internal state after replacement
                    self._post_chain_replace_update(
                        # Pass the actual new chain
                        best_chain_info['chain_obj_list'])
                else:
                    # This might happen if the chain became invalid between the check and now,
                    # or if the replace_chain logic has stricter checks failed earlier.
                    print(f"NODE {
                          self.node_id}: Chain replacement failed (validation or length check in replace_chain). Keeping current chain.")
            else:
                # Current chain is still the best known chain
                print(
                    f"NODE {self.node_id}: Current chain remains the best known chain.")

    def get_chain_copy(self):
        """Provides a deep copy of the node's current blockchain list (for simulation sync)."""
        # --- Simulation Helper ---
        with self.lock:
            # Return a deep copy to prevent external modification of the node's actual chain
            return copy.deepcopy(self.blockchain.chain)

    def _post_chain_replace_update(self, new_chain_objects):
        """
        Updates the node's internal state after its blockchain has been successfully replaced.
        Must be called *after* self.blockchain.chain is updated and *while holding the lock*.
        """
        print(f"NODE {self.node_id}: Updating state after chain replacement...")
        # 1. Update the set of known transaction hashes based on the new chain reality
        #    This call clears and rebuilds known_tx_hashes from the new self.blockchain.chain
        #    and also re-adds pending transactions.
        self._update_known_tx_hashes()

        # 2. Reconcile Pending Transactions (Crucial & Complex Task)
        #    - Transactions in the pending pool might now be confirmed in the new chain. Remove them.
        #    - Transactions that were confirmed in the *old* chain but are *not* in the *new* chain (due to fork rollback)
        #      should ideally be added *back* to the pending pool if they are still valid.
        # --- Simplified Reconciliation (Remove newly confirmed txs) ---
        newly_confirmed_hashes = set()
        for block in new_chain_objects:  # Iterate the actual new chain
            for tx_data in block.transactions:
                try:
                    tx_temp = Transaction.from_dict(tx_data)
                    newly_confirmed_hashes.add(tx_temp.calculate_hash())
                except Exception:
                    continue  # Skip malformed tx

        removed_from_pending_count = 0
        current_pending_hashes = list(
            self.pending_transactions.keys())  # Iterate copy
        for tx_hash in current_pending_hashes:
            if tx_hash in newly_confirmed_hashes:
                if tx_hash in self.pending_transactions:  # Double check existence
                    del self.pending_transactions[tx_hash]
                    removed_from_pending_count += 1
                # Ensure it remains in known_tx_hashes (handled by _update_known_tx_hashes)

        print(f"NODE {self.node_id}: Reconciled pending pool. Removed {
              removed_from_pending_count} newly confirmed txs.")
        # Note: This simplified version doesn't automatically re-add transactions from the invalidated part of the old fork.
        # A full implementation would require comparing the old and new chains block by block.

        # 3. Save the new chain state to the node's specific file
        self.save_blockchain()

        # 4. Process orphans based on the new chain tip
        #    The new tip might unlock previously stored orphans.
        new_tip_hash = new_chain_objects[-1].hash if new_chain_objects else None
        if new_tip_hash:
            # Check based on the actual new tip
            self._process_orphans(new_tip_hash)

        print(
            f"NODE {self.node_id}: State update after chain replacement complete.")

    def _remove_txs_from_pending(self, block):
        """Removes transactions included in a newly added block from the pending pool."""
        # Assumes self.lock is already held by the caller (e.g., handle_received_block, _mine_loop)
        removed_count = 0
        tx_hashes_in_block = set()
        for tx_data in block.transactions:
            try:
                # Recreate object to get the hash consistently
                tx_obj_temp = Transaction.from_dict(tx_data)
                tx_hash = tx_obj_temp.calculate_hash()
                # Keep track of all hashes in block
                tx_hashes_in_block.add(tx_hash)

                # Remove from pending pool if it exists there
                if tx_hash in self.pending_transactions:
                    del self.pending_transactions[tx_hash]
                    removed_count += 1

                # Ensure this transaction hash is marked as known (covers reward tx etc.)
                self.known_tx_hashes.add(tx_hash)
            except Exception as e:
                print(f"NODE {self.node_id}: WARNING - Error processing tx data in block {
                      block.index} for pending removal: {e}")
                continue  # Skip malformed tx data
        return removed_count

    # --- Mining Logic ---

    def _mine_loop(self):
        """Background thread target method for mining new blocks."""
        print(f"NODE {self.node_id}: Mining thread started.")
        while not self.stop_event.is_set():
            pending_tx_list_snapshot = []
            miner_address = self.wallet.address  # Get miner's address once

            if not miner_address:
                print(f"NODE {
                      self.node_id}: ERROR - Cannot mine, wallet address is missing or invalid.")
                # Wait longer before retrying if wallet is broken
                time.sleep(10)
                continue

            with self.lock:  # Lock to safely access pending transactions and latest block
                # Create a snapshot of transactions to include in the potential new block
                # Sort for deterministic block creation (optional but good practice)
                pending_items = sorted(
                    self.pending_transactions.items())  # Sort by tx hash
                # Filter out invalid transactions again? Or assume valid? Assume valid for now.
                pending_tx_list_snapshot = [
                    tx for tx_hash, tx in pending_items]

                # Get the latest block info needed to build the new block *at this moment*
                latest_block = self.blockchain.get_latest_block()
                intended_prev_block_hash = latest_block.hash
                intended_prev_block_index = latest_block.index

                # Prepare log message inside lock to capture state accurately
                num_pending = len(pending_tx_list_snapshot)
                print(f"NODE {self.node_id}: Preparing to mine block {intended_prev_block_index +
                      1} on top of {intended_prev_block_hash[:8]} (found {num_pending} pending txs)...")

            # --- Mining Operation (Proof-of-Work) ---
            # This happens *outside* the lock, allowing other threads (e.g., processing) to run
            new_block = None
            try:
                # The create_new_block method handles adding the reward transaction
                # and performing the Proof-of-Work calculation.
                new_block = self.blockchain.create_new_block(pending_tx_list_snapshot, miner_address)
                # Check if stop event was set during potentially long mining process
                if self.stop_event.is_set():
                    print(
                        f"NODE {self.node_id}: Mining interrupted by stop signal.")
                    break  # Exit the loop

                if new_block:
                    print(f"NODE {self.node_id}: Successfully mined Block {
                          new_block.index} [{new_block.hash[:8]}]!")
                else:
                    # Should not happen if create_new_block doesn't raise exceptions
                    print(
                        f"NODE {self.node_id}: Mining attempt failed, create_new_block returned None.")
                    time.sleep(1)  # Pause before retrying
                    continue

            except Exception as e:
                print(
                    f"NODE {self.node_id}: ERROR during block creation/mining: {e}")
                import traceback
                traceback.print_exc()  # Essential for debugging mining errors
                time.sleep(2)  # Pause after error before retrying
                continue  # Go to the next iteration of the while loop

            # --- Post-Mining: Add Block to Chain (Critical Section) ---
            block_added_successfully = False
            with self.lock:  # Re-acquire lock to check chain state and add the block
                # CRITICAL CHECK: Has the chain tip changed while we were mining?
                current_latest_block = self.blockchain.get_latest_block()
                # Compare the intended parent hash with the hash in the mined block
                if new_block.previous_hash != intended_prev_block_hash:
                    print(f"NODE {self.node_id}: Mined block {new_block.index} is STALE. "
                          f"Parent mismatch ({new_block.previous_hash[:8]} != intended {
                        intended_prev_block_hash[:8]}) "
                        f"Current tip is {current_latest_block.hash[:8]}. Discarding.")
                    # Don't add the block, just continue to the next mining attempt
                    continue  # Skip adding and broadcasting

                # If chain state is consistent, attempt to add the newly mined block
                # The add_block method performs final validation (index, prev_hash, tx validity)
                # Returns True if added successfully
                if self.blockchain.add_block(new_block):
                    print(
                        f"NODE {self.node_id}: Added self-mined block {new_block.index} to chain.")
                    # Block added successfully, remove included TXs from pending pool
                    removed_count = self._remove_txs_from_pending(new_block)
                    print(f"NODE {self.node_id}: Removed {
                          removed_count} confirmed txs from pool.")
                    # Save the updated chain state
                    self.save_blockchain()
                    # Check if this new block unlocks orphans
                    self._process_orphans(new_block.hash)
                    block_added_successfully = True
                else:
                    # This case should be rare if the stale check above works, but handles failures in add_block validation
                    print(f"NODE {self.node_id}: Failed to add self-mined block {
                          new_block.index} via add_block (failed validation?). Discarding.")
                    continue  # Skip broadcasting

            # --- Broadcast the newly mined block (if added successfully) ---
            # Do this *outside* the lock to avoid holding it during network operations
            if block_added_successfully:
                self.broadcast_block(new_block)

            # Small pause helps prevent overly tight mining loops if difficulty is very low
            # and allows other nodes' messages to be processed.
            time.sleep(0.1)  # Adjust as needed

        print(f"NODE {self.node_id}: Mining thread finished.")

    # --- Utility Methods ---
    def create_and_broadcast_transaction(self, receiver_address, amount, fee=DEFAULT_FEE):
        """Creates a signed transaction from this node's wallet and broadcasts it."""
        print(f"\nNODE {self.node_id}: Creating transaction: {
              amount} (+{fee} fee) -> {receiver_address[:8]}...")

        # Ensure amount and fee are positive
        if amount <= 0 or fee < 0:
            print(f"NODE {
                  self.node_id}: Transaction FAILED - amount ({amount}) and fee ({fee}) must be positive.")
            return None

        sender_pub_key_pem = self.wallet.get_public_key_pem()
        sender_address = self.wallet.address

        if not sender_pub_key_pem or not sender_address:
            print(
                f"NODE {self.node_id}: Transaction FAILED - Wallet keys/address missing.")
            return None

        # Check balance within lock
        with self.lock:
            balance = self.blockchain.get_balance(sender_address)
            required = amount + fee
            if balance < required:
                print(f"NODE {
                      self.node_id}: Transaction FAILED - insufficient funds ({balance: .4f} < {required: .4f}).")
                return None

        # Create transaction object (outside lock is fine)
        tx = Transaction(
            sender_public_key_pem=sender_pub_key_pem,
            receiver_address=receiver_address,
            amount=amount,
            fee=fee,
        )
        # Sign the transaction (requires private key access, handled by Wallet)
        try:
            tx.sign(self.wallet)
            if not tx.signature:
                raise ValueError("Signing failed, signature is missing.")
        except Exception as e:
            print(
                f"NODE {self.node_id}: Transaction FAILED - could not sign transaction: {e}")
            return None

        # Add to own pending pool first (requires lock)
        tx_hash = tx.calculate_hash()
        with self.lock:
            if tx_hash not in self.known_tx_hashes:
                self.pending_transactions[tx_hash] = tx
                self.known_tx_hashes.add(tx_hash)
                print(f"NODE {self.node_id}: Added own Tx {
                      tx_hash[:8]} to pending pool.")
            else:
                # This might happen if somehow the transaction was already received/processed
                print(f"NODE {self.node_id}: Own Tx {
                      tx_hash[:8]} was already known?. Ignoring add to pool.")

        # Broadcast the transaction (outside lock)
        self.broadcast_transaction(tx)
        print(f"NODE {self.node_id}: Transaction {
              tx_hash[:8]} created and broadcast successfully.")
        return tx

    def get_balance(self):
        """Gets the balance for this node's wallet address from its view of the blockchain."""
        with self.lock:  # Accessing blockchain requires lock
            return self.blockchain.get_balance(self.wallet.address)

    def print_status(self):
        """Prints a summary of the node's current status."""
        with self.lock:  # Acquire lock to get consistent snapshot of state
            try:
                chain_len = len(self.blockchain.chain)
                latest_block = self.blockchain.get_latest_block()
                latest_block_hash = latest_block.hash if latest_block else "N/A"
                latest_block_index = latest_block.index if latest_block else -1
                balance = self.blockchain.get_balance(self.wallet.address)
                pending_tx_count = len(self.pending_transactions)
                orphan_count = len(self.orphan_blocks)
                peer_ids = list(self.peers.keys())

                print(f"\n--- STATUS Node {self.node_id} ---")
                print(f"  Address:          {self.wallet.address[:12]}...")
                print(f"  Balance:          {balance: .4f}")  # Format balance
                print(f"  Chain Length:     {chain_len}")
                print(f"  Latest Block Idx: {latest_block_index}")
                print(f"  Latest Block Hash: {latest_block_hash[:12]}...")
                print(f"  Pending Txs:      {pending_tx_count}")
                print(f"  Orphan Blocks:    {orphan_count}")
                print(f"  Known Peers:      {peer_ids}")
                print(f"--------------------------")

            except Exception as e:
                print(f"NODE {self.node_id}: Error printing status: {e}")


# --- Main Simulation Script ---
if __name__ == "__main__":
    # --- Simulation Parameters ---
    NUM_NODES = 3
    SIMULATION_TIME = 60  # Run for 60 seconds
    TX_INTERVAL = 5       # Try to generate a transaction every 5 seconds
    DIFFICULTY = 5        # Must match Blockchain class
    SYNC_INTERVAL = 10    # Trigger proactive chain sync every 10 seconds

    # --- Clean up previous run data (Optional but Recommended) ---
    print("--- Cleaning up previous blockchain data ---")
    # Remove the shared file if it exists
    # Use the constant from Blockchain class
    shared_chain_file = Blockchain.DEFAULT_CHAIN_FILE
    if os.path.exists(shared_chain_file):
        try:
            os.remove(shared_chain_file)
            print(f"Removed shared file: {shared_chain_file}")
        except OSError as e:
            print(f"Error removing {shared_chain_file}: {e}")

    # Remove node-specific files and the blockchain data directory
    if os.path.exists(BLOCKCHAIN_DIR):
        import shutil
        try:
            shutil.rmtree(BLOCKCHAIN_DIR)
            print(f"Removed directory: {BLOCKCHAIN_DIR}")
        except OSError as e:
            print(f"Error removing directory {BLOCKCHAIN_DIR}: {e}")

    # Optionally remove wallet files to generate fresh keys each run
    # Be careful if you want wallets to persist across runs
    # if os.path.exists(WALLET_DIR):
    #    import shutil
    #    try:
    #        shutil.rmtree(WALLET_DIR)
    #        print(f"Removed directory: {WALLET_DIR}")
    #    except OSError as e:
    #         print(f"Error removing directory {WALLET_DIR}: {e}")

    print("--- Cleanup Complete ---")
    time.sleep(1)  # Small pause after cleanup

    # --- Reset Node Registry (Important for reruns in same process/notebook) ---
    NODE_REGISTRY.clear()
    nodes = []

    try:
        # --- Setup ---
        print("\n--- Creating P2P Nodes ---")
        # Ensure directories exist *before* creating nodes that might need them
        ensure_dir(WALLET_DIR)
        # Crucial: ensure this exists before Blockchain() might try saving
        ensure_dir(BLOCKCHAIN_DIR)

        # Create Node instances
        for i in range(NUM_NODES):
            node_id = f"node_{i}"
            # Create the node. It initializes wallet & blockchain (loads/creates genesis)
            # Uses global DIFFICULTY implicitly via Blockchain class
            node = P2PNode(node_id=node_id)
            nodes.append(node)
            # Register the node globally for peer discovery (simulation specific)
            with NODE_REGISTRY_LOCK:
                NODE_REGISTRY[node_id] = node
        print(f"Created and registered {NUM_NODES} nodes.")

        # --- Initial State Verification ---
        print("\n--- Initial Node State Check ---")
        initial_genesis_hash = None
        all_match = True
        for i, node in enumerate(nodes):
            with node.lock:  # Access blockchain safely
                if node.blockchain and node.blockchain.chain:  # Check if chain exists and is not empty
                    current_genesis_hash = node.blockchain.chain[0].hash
                    print(f"Node {node.node_id}: Genesis Hash = {
                          current_genesis_hash[:12]}..., Chain Length = {len(node.blockchain.chain)}")
                    if i == 0:
                        initial_genesis_hash = current_genesis_hash
                    elif current_genesis_hash != initial_genesis_hash:
                        print(f"ERROR: Node {node.node_id} Genesis mismatch!")
                        all_match = False
                    # Also check if loading specific chain resulted in different lengths initially
                    if len(node.blockchain.chain) != 1 and i > 0 and len(node.blockchain.chain) != len(nodes[0].blockchain.chain):
                        print(f"WARN: Node {node.node_id} has different initial chain length ({len(
                            node.blockchain.chain)}) than Node 0 ({len(nodes[0].blockchain.chain)}). (Might be ok if resuming)")
                        # Decide if this is an error state? For clean start, it might be.

                else:
                    print(
                        f"Node {node.node_id}: ERROR - Chain is missing or empty!")
                    all_match = False
                    # break # Exit loop early if critical error found

        if all_match and initial_genesis_hash:
            print("SUCCESS: All nodes initialized with the same Genesis Block.")
        else:
            print("CRITICAL ERROR: Nodes initialized with different or missing Genesis Blocks! Aborting simulation.")
            # Cleanly stop any partially started nodes if necessary? (Not needed here as threads not started yet)
            exit(1)  # Exit the script

        # --- Start Node Threads ---
        print("\n--- Starting Nodes ---")
        for node in nodes:
            node.start()  # This connects peers and starts mining/processing threads

        print("\n--- Running Simulation ---")
        start_time = time.time()
        last_tx_time = start_time
        last_sync_time = start_time  # Initialize sync timer
        tx_counter = 0

        # --- Main Simulation Loop ---
        while time.time() - start_time < SIMULATION_TIME:
            current_time = time.time()

            # --- Periodic Transaction Generation ---
            if current_time - last_tx_time >= TX_INTERVAL:
                sender_node = random.choice(nodes)
                # Check if sender has enough balance to at least cover the default fee + a minimal amount
                can_send = False
                min_send_amount = 1  # Minimum amount to send
                required_for_tx = min_send_amount + DEFAULT_FEE
                # Get balance requires lock, do it briefly
                sender_balance = sender_node.get_balance()  # Use the helper method

                if sender_balance >= required_for_tx:
                    can_send = True
                    # print(f"SIM: Node {sender_node.node_id} has sufficient balance ({sender_balance}) to send.") # Debug log
                # else:
                    # print(f"SIM: Node {sender_node.node_id} has insufficient balance ({sender_balance}) to send {required_for_tx}.") # Debug log

                if can_send:
                    # Select a random receiver (different from sender)
                    receiver_node = random.choice(
                        [n for n in nodes if n != sender_node])
                    # Ensure receiver address is valid (wallet initialized)
                    if receiver_node.wallet.address:
                        # Generate random amount and fee
                        # Ensure amount doesn't exceed balance minus fee
                        max_possible_amount = int(sender_balance - DEFAULT_FEE)
                        if max_possible_amount >= min_send_amount:  # Check if can afford min amount + fee
                            amount = random.randint(min_send_amount, max(
                                min_send_amount, max_possible_amount))  # Send between 1 and max possible
                            fee = DEFAULT_FEE  # Use default fee for simplicity

                            print(f"\nSIM: Attempting Tx {tx_counter+1} from {
                                  sender_node.node_id} ({sender_balance: .2f}) to {receiver_node.node_id}...")
                            # create_and_broadcast handles internal balance check again for the specific amount+fee
                            tx_result = sender_node.create_and_broadcast_transaction(receiver_node.wallet.address, amount, fee)
                            if tx_result:
                                tx_counter += 1
                                print(f"SIM: Tx {tx_counter} broadcast initiated by {
                                      sender_node.node_id}.")
                            # else: Transaction failed (e.g., race condition on balance), message printed inside method
                        else:
                            # This case should be rare due to the initial check, but handles edge cases
                            print(f"SIM: Skipping Tx - Node {sender_node.node_id} balance ({
                                  sender_balance}) too low for min amount + fee.")
                    else:
                        print(
                            f"SIM: Skipping Tx - Receiver {receiver_node.node_id} wallet/address not ready.")
                else:
                    # Only print occasionally if skipping due to low balance to avoid spamming logs
                    if random.random() < 0.1:  # Print approx 10% of the time
                        print(f"SIM: Skipping Tx attempt from {sender_node.node_id} (insufficient balance: {
                              sender_balance: .2f} < {required_for_tx: .2f}).")

                # Update time regardless of success/failure to maintain interval
                last_tx_time = current_time

            # --- Periodic Chain Synchronization ---
            if current_time - last_sync_time >= SYNC_INTERVAL:
                print(
                    f"\nSIM: Triggering Periodic Chain Sync (Interval: {SYNC_INTERVAL}s)...")
                # Randomly pick a node to initiate the sync check with its peers
                sync_node = random.choice(nodes)
                print(f"SIM: Node {
                      sync_node.node_id} will initiate sync check.")
                try:
                    # This node will ask others for their chains and potentially replace its own if a longer valid one is found
                    sync_node.request_chain_from_peers()
                except Exception as e:
                    print(f"SIM: Error during periodic sync initiated by {
                          sync_node.node_id}: {e}")
                    import traceback
                    traceback.print_exc()  # Print stack trace for sync errors
                last_sync_time = current_time

            # --- Optional: Check if threads are still alive (for debugging) ---
            # for n in nodes:
            #     if not n.mining_thread.is_alive() or not n.processing_thread.is_alive():
            #         print(f"FATAL WARN: Thread died unexpectedly for node {n.node_id}! Simulation may be compromised.")
            #         # Consider stopping simulation or attempting restart?

            # Main loop delay - prevents this loop from consuming 100% CPU
            time.sleep(0.5)  # Check conditions twice per second

        print(f"\n--- Simulation Time ({SIMULATION_TIME}s) Ended ---")

    except KeyboardInterrupt:
        print("\n--- Simulation Interrupted by User (Ctrl+C) ---")
        # Perform cleanup in finally block

    finally:
        # --- Stop Nodes Gracefully ---
        print("\n--- Stopping Nodes ---")
        for node in nodes:
            if node:  # Check if node object exists
                try:
                    node.stop()  # Signal threads and wait for them
                except Exception as e:
                    print(f"Error stopping node {node.node_id}: {e}")

        print("\n--- Final Node Status ---")
        # Give threads a moment to finish printing messages after stop signal
        time.sleep(2)

        # --- Final State Analysis ---
        all_chains_match = True
        first_chain_hash_list = None
        final_states = {}  # Store final state {node_id: {state_info}}

        for i, node in enumerate(nodes):
            if node:  # Check node exists
                node.print_status()  # Print detailed status first
                with node.lock:  # Access node state safely
                    # Store key state info for comparison
                    current_chain_hashes = [
                        b.hash for b in node.blockchain.chain] if node.blockchain else []
                    final_states[node.node_id] = {
                        'length': len(node.blockchain.chain) if node.blockchain else 0,
                        'tip': current_chain_hashes[-1] if current_chain_hashes else None,
                        'hashes': current_chain_hashes,  # Store the list of hashes
                        'balance': node.blockchain.get_balance(node.wallet.address) if node.blockchain else 0,
                        'orphans': len(node.orphan_blocks),
                        'pending': len(node.pending_transactions)
                    }

                    # Use the first node's chain as the reference for comparison
                    if i == 0:
                        first_chain_hash_list = current_chain_hashes
                    # Compare the entire list of hashes for exact match
                    elif current_chain_hashes != first_chain_hash_list:
                        all_chains_match = False
            else:
                print(f"Node object at index {i} doesn't exist or was None.")

        print("\n--- Chain Consistency Check ---")
        if final_states:
            first_node_id = list(final_states.keys())[0]
            ref_state = final_states[first_node_id]
            print(f"Reference Chain (from {first_node_id}): Length={ref_state['length']}, Tip={
                  ref_state['tip'][:12] if ref_state['tip'] else 'N/A'}...")

            for node_id, state in final_states.items():
                if node_id == first_node_id:
                    continue  # Skip comparing ref to itself
                # Compare hash lists for definitive match/mismatch
                match_status = "MATCH" if state['hashes'] == ref_state['hashes'] else "MISMATCH"
                print(f"  Node {node_id}: Length={state['length']}, Tip={
                      state['tip'][:12] if state['tip'] else 'N/A'}... -> {match_status}")
                print(f"    (Balance: {state['balance']: .2f}, Orphans: {
                      state['orphans']}, Pending: {state['pending']})")
                # Optionally print full hash list difference if mismatch for debugging
                # if match_status == "MISMATCH":
                #     # Find first differing block? Compare lengths?
                #     pass # Add detailed diff logic if needed

            if all_chains_match:
                print(
                    "\n+++ SUCCESS: All nodes converged to the same blockchain state. +++")
            else:
                print(
                    "\n--- WARNING: Nodes have different blockchain states. Consensus NOT fully reached. ---")
                print(
                    "  (This can happen due to network delays, concurrent mining, or insufficient sync time.)")
        else:
            print(
                "Could not perform final chain consistency check (no final node states captured).")

        print("\n--- Simulation Complete ---")
