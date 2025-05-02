import numpy as np
from collections import defaultdict, deque
import heapq
from typing import Dict, List, Tuple, Optional
from enum import Enum
from HBM import HBMMemory
class OperationType(Enum):
    READ = "read"
    WRITE = "write"
    FETCH = "fetch"
    CONSUME = "consume"

class CacheEntry:
    def __init__(self, data: any, rrpv: int = 2, priority: int = 0):
        self.data = data
        self.rrpv = rrpv  # Re-reference prediction value (SRRIP) - used for tie-breaking
        self.priority = priority  # Priority for replacement (5-bit counter for 32 PEs)
        self.valid = True
        self.dirty = False
        self.tag = None
        self.fetched = False

class FiberCache:
    def __init__(self, num_banks: int = 48, bank_size: int = 64*1024, 
             associativity: int = 16, block_size: int = 64):
        self.num_banks = num_banks
        self.bank_size = bank_size
        self.associativity = associativity
        self.block_size = block_size
        self.num_sets = bank_size // (block_size * associativity)
        
        # Priority management
        self.priority_bits = 5  # 5-bit counter for 32 PEs
        self.max_priority = (1 << self.priority_bits) - 1
        
        # Initialize cache banks
        self.banks = []
        for _ in range(num_banks):
            bank = []
            for _ in range(self.num_sets):
                cache_set = [None] * self.associativity
                bank.append(cache_set)
            self.banks.append(bank)
        
        # Bank conflict management
        self.bank_busy = [False] * num_banks  # Track busy banks
        self.bank_queue = [deque() for _ in range(num_banks)]  # Queue per bank
        self.banks_to_release = []
        
        # Queues for operations
        self.fetch_queue = deque()
        self.read_queue = deque()
        
        # Outstanding requests tracking
        self.outstanding_requests = {}
        self.request_latency = {}
        
        # Pending queue for memory operations
        self.pending_queue = deque()
        
        # HBM memory instance
        self.memory = HBMMemory(num_channels=16, channel_bandwidth_gbps=8.0, 
                            bus_width_bits=64, latency_cycles=100)
        
        # Statistics
        self.stats = {
            'onchip_reads': 0,
            'onchip_writes': 0,
            'offchip_reads': 0,
            'offchip_writes': 0,
            'read_misses': 0,
            'write_misses': 0,
            'consume_misses': 0,
            'fetches': 0,
            'consumes': 0,
            'read_hits': 0,
            'write_hits': 0,
            'consume_hits': 0,
            'onchip_bandwidth': 0,
            'offchip_bandwidth': 0,
            'bank_conflicts': 0
        }
        
        # Memory latency simulation
        self.memory_latency = 100  # cycles
        self.bus_width = 8  # bytes
        
    def _check_bank_conflict(self, bank_idx: int, operation: dict) -> bool:
        """Check if bank is busy and queue operation if needed"""
        if self.bank_busy[bank_idx]:
            # Bank is busy, queue the operation
            print("Hi")
            self.bank_queue[bank_idx].append(operation)
            self.stats['bank_conflicts'] += 1
            return True
        else:
            # Bank is available, mark it as busy
            self.bank_busy[bank_idx] = True
            return False
    
    def _release_bank(self, bank_idx: int):
        """Release bank"""
        #print("Hi")
        #self.bank_busy[bank_idx] = False
        self.banks_to_release.append(bank_idx)
    
    def _process_bank_operation(self, bank_idx: int, operation: dict):
        """Process a bank operation"""
        op_type = operation['type']
        
        if op_type == OperationType.READ:
            return self._process_read_operation(bank_idx, operation)
        elif op_type == OperationType.WRITE:
            return self._process_write_operation(bank_idx, operation)
        elif op_type == OperationType.FETCH:
            return self._process_fetch_operation(bank_idx, operation)
        elif op_type == OperationType.CONSUME:
            return self._process_consume_operation(bank_idx, operation)
    
    def _get_bank_and_set(self, address: int) -> Tuple[int, int]:
        """Calculate bank and set index from address"""
        # Simple address mapping to handle bank conflicts
        block_addr = address // self.block_size
        bank_idx = block_addr % self.num_banks
        set_idx = (block_addr // self.num_banks) % self.num_sets
        return bank_idx, set_idx
    
    def _find_victim(self, bank_idx: int, set_idx: int) -> int:
        """Find victim using priority and SRRIP as tie-breaker"""
        cache_set = self.banks[bank_idx][set_idx]
        
        # First, check for invalid entries
        for i, entry in enumerate(cache_set):
            if entry is None or not entry.valid:
                return i
        
        # Find entry with lowest priority
        min_priority = self.max_priority + 1
        candidates = []
        
        for i, entry in enumerate(cache_set):
            if entry and entry.valid:
                if entry.priority < min_priority:
                    min_priority = entry.priority
                    candidates = [(i, entry.rrpv)]
                elif entry.priority == min_priority:
                    candidates.append((i, entry.rrpv))
        
        # If multiple candidates with same priority, use SRRIP
        while True:
            # Find candidate with highest RRPV
            max_rrpv = -1
            victim_idx = -1
            
            for idx, rrpv in candidates:
                if rrpv > max_rrpv:
                    max_rrpv = rrpv
                    victim_idx = idx
            
            # If found entry with RRPV=3, use it as victim
            if max_rrpv == 3:
                return victim_idx
            
            # If max RRPV < 3, increment all candidate RRPVs
            new_candidates = []
            for idx, rrpv in candidates:
                entry = cache_set[idx]
                entry.rrpv = min(3, entry.rrpv + 1)
                new_candidates.append((idx, entry.rrpv))
            candidates = new_candidates
    
    def _update_priority(self, entry: CacheEntry, operation: OperationType):
        """Update priority based on operation type"""
        if operation == OperationType.FETCH:
            # Fetch increments priority
            entry.priority = min(self.max_priority, entry.priority + 1)
        elif operation == OperationType.READ:
            # Read decrements priority
            entry.priority = max(0, entry.priority - 1)
        # Write and consume operations don't change priority
    
    def _update_rrpv(self, entry: CacheEntry):
        """Update RRPV on access (hit)"""
        # According to SRRIP paper, on hit, RRPV should be set to 0
        entry.rrpv = 0
    
    def read(self, address: int) -> Tuple[Optional[any], bool]:
        """Read operation - returns (data, valid)"""
        # Align address to block boundary
        block_address = (address // self.block_size) * self.block_size
        bank_idx, set_idx = self._get_bank_and_set(block_address)
        
        # Check for bank conflict
        if self.bank_busy[bank_idx]:
            # Check if this operation is already pending
            for op in self.pending_queue:
                if op['type'] == OperationType.READ and op['address'] == block_address:
                    return None, False  # Already pending, ignore duplicate
            
            # Bank is busy, add to pending queue
            operation = {
                'type': OperationType.READ,
                'address': block_address,
                'set_idx': set_idx
            }
            self.pending_queue.append(operation)
            return None, False
        
        # Bank is available, mark it as busy
        self.bank_busy[bank_idx] = True
        
        # Check cache
        cache_set = self.banks[bank_idx][set_idx]
        
        # Check for hit
        for entry in cache_set:
            if entry and entry.valid and entry.tag == block_address:
                self._update_rrpv(entry)
                self._update_priority(entry, OperationType.READ)
                self.stats['read_hits'] += 1
                self.stats['onchip_reads'] += 1
                self.stats['onchip_bandwidth'] += self.block_size
                #self.bank_busy[bank_idx] = False  # Release bank
                self.banks_to_release.append(bank_idx)
                
                # Return the requested data from the block
                # In a real implementation, we would extract the specific word
                # For simulation, we'll return the entire block
                return entry.data, True
        
        # Cache miss - check if there's already an outstanding request
        for request in self.outstanding_requests.values():
            if request['type'] == OperationType.READ and request['address'] == block_address:
                #self.bank_busy[bank_idx] = False  # Release bank
                self.banks_to_release.append(bank_idx)
                return None, False  # Already outstanding, ignore duplicate
        
        # Cache miss - invalidate and fetch from memory
        self.stats['read_misses'] += 1
        self.stats['offchip_reads'] += 1
        self.stats['offchip_bandwidth'] += self.block_size
        
        # Find victim and invalidate
        victim_idx = self._find_victim(bank_idx, set_idx)
        if cache_set[victim_idx] and cache_set[victim_idx].valid:
            if cache_set[victim_idx].dirty:
                # Writeback dirty data to memory
                wb_address = cache_set[victim_idx].tag
                wb_data = cache_set[victim_idx].data
                self.memory.write(wb_address, wb_data, self.block_size)
                self.stats['offchip_writes'] += 1
                self.stats['offchip_bandwidth'] += self.block_size
            
            # Invalidate victim
            cache_set[victim_idx].valid = False
        
        # Issue memory read request for entire block
        mem_request_id = self.memory.read(block_address, self.block_size)
        self.outstanding_requests[mem_request_id] = {
            'type': OperationType.READ,
            'address': block_address
        }
        
        #self.bank_busy[bank_idx] = False  # Release bank
        self.banks_to_release.append(bank_idx)
        return None, False
    
    def _process_read_operation(self, bank_idx: int, operation: dict) -> Tuple[Optional[any], bool]:
        """Process a read operation"""
        address = operation['address']
        set_idx = operation['set_idx']
        cache_set = self.banks[bank_idx][set_idx]
        
        # Check for hit
        for entry in cache_set:
            if entry and entry.valid and entry.tag == address:
                self._update_rrpv(entry)
                self._update_priority(entry, OperationType.READ)
                self.stats['read_hits'] += 1
                self.stats['onchip_reads'] += 1
                self.stats['onchip_bandwidth'] += self.block_size
                self._release_bank(bank_idx)
                return entry.data, True
        
        # Cache miss
        self.stats['read_misses'] += 1
        self.stats['offchip_reads'] += 1
        self.stats['offchip_bandwidth'] += self.block_size
        
        # Initiate memory request
        request_id = f"read_{address}_{self.stats['offchip_reads']}"
        self.outstanding_requests[request_id] = {
            'type': OperationType.READ,
            'address': address,
            'data': None
        }
        self.request_latency[request_id] = self.memory_latency
        #print("Hi")
        self._release_bank(bank_idx)
        return None, False
    
    def write(self, address: int, data: any) -> bool:
        """Write operation - returns success status"""
        bank_idx, set_idx = self._get_bank_and_set(address)
        
        # Check for bank conflict
        if self.bank_busy[bank_idx]:
            # Bank is busy, add to pending queue
            operation = {
                'type': OperationType.WRITE,
                'address': address,
                'data': data,
                'set_idx': set_idx
            }
            self.pending_queue.append(operation)
            return False
        
        # Bank is available, mark it as busy
        self.bank_busy[bank_idx] = True
        
        # Process write operation
        cache_set = self.banks[bank_idx][set_idx]
        
        # Check for hit
        for entry in cache_set:
            if entry and entry.valid and entry.tag == address:
                self._update_rrpv(entry)
                entry.data = data
                entry.dirty = True
                self.stats['write_hits'] += 1
                self.stats['onchip_writes'] += 1
                self.stats['onchip_bandwidth'] += self.block_size
                #self.bank_busy[bank_idx] = False  # Release bank
                self.banks_to_release.append(bank_idx)
                return True
        
        # Cache miss - find victim
        victim_idx = self._find_victim(bank_idx, set_idx)
        
        if cache_set[victim_idx] and cache_set[victim_idx].valid and cache_set[victim_idx].dirty:
            # Write back dirty data
            wb_address = cache_set[victim_idx].tag
            wb_data = cache_set[victim_idx].data
            self.memory.write(wb_address, wb_data, self.block_size)
            self.stats['offchip_writes'] += 1
            self.stats['offchip_bandwidth'] += self.block_size
        
        # Allocate new entry
        new_entry = CacheEntry(data, rrpv=2, priority=0)
        new_entry.tag = address
        new_entry.dirty = True
        cache_set[victim_idx] = new_entry
        
        self.stats['write_misses'] += 1
        self.stats['onchip_writes'] += 1
        self.stats['onchip_bandwidth'] += self.block_size
        
        #self.bank_busy[bank_idx] = False  # Release bank
        self.banks_to_release.append(bank_idx)
        return True
    
    def _process_write_operation(self, bank_idx: int, operation: dict) -> bool:
        """Process a write operation"""
        address = operation['address']
        data = operation['data']
        set_idx = operation['set_idx']
        cache_set = self.banks[bank_idx][set_idx]
        
        # Check for hit
        for entry in cache_set:
            if entry and entry.valid and entry.tag == address:
                self._update_rrpv(entry)
                entry.data = data
                entry.dirty = True
                self.stats['write_hits'] += 1
                self.stats['onchip_writes'] += 1
                self.stats['onchip_bandwidth'] += self.block_size
                self._release_bank(bank_idx)
                return True
        
        # Cache miss - find victim
        victim_idx = self._find_victim(bank_idx, set_idx)
        
        if cache_set[victim_idx] and cache_set[victim_idx].valid and cache_set[victim_idx].dirty:
            # Write back dirty data
            self.stats['offchip_writes'] += 1
            self.stats['offchip_bandwidth'] += self.block_size
        
        # Allocate new entry with RRPV=2 (long re-reference interval) and priority=0
        new_entry = CacheEntry(data, rrpv=2, priority=0)
        new_entry.tag = address
        new_entry.dirty = True
        cache_set[victim_idx] = new_entry
        
        self.stats['write_misses'] += 1
        self.stats['onchip_writes'] += 1
        self.stats['onchip_bandwidth'] += self.block_size
        
        self._release_bank(bank_idx)
        return True
    
    def fetch(self, address: int) -> bool:
        """Fetch operation - returns success status"""
        bank_idx, set_idx = self._get_bank_and_set(address)
        
        # Check for bank conflict
        if self.bank_busy[bank_idx]:
            # Check if this operation is already pending
            for op in self.pending_queue:
                if op['type'] == OperationType.FETCH and op['address'] == address:
                    return False  # Already pending, ignore duplicate
            
            # Bank is busy, add to pending queue
            operation = {
                'type': OperationType.FETCH,
                'address': address,
                'set_idx': set_idx
            }
            self.pending_queue.append(operation)
            return False
        
        # Bank is available, mark it as busy
        self.bank_busy[bank_idx] = True
        
        # Check cache
        cache_set = self.banks[bank_idx][set_idx]
        
        # Check if already in cache
        for entry in cache_set:
            if entry and entry.valid and entry.tag == address:
                self._update_priority(entry, OperationType.FETCH)
                #self.bank_busy[bank_idx] = False  # Release bank
                self.banks_to_release.append(bank_idx)
                return True
        
        # Check if there's already an outstanding request
        for request in self.outstanding_requests.values():
            if request['type'] == OperationType.FETCH and request['address'] == address:
                #self.bank_busy[bank_idx] = False  # Release bank
                self.banks_to_release.append(bank_idx)
                return True  # Already outstanding, ignore duplicate
        
        # Not in cache, issue fetch
        self.stats['fetches'] += 1
        
        # Find victim and invalidate
        victim_idx = self._find_victim(bank_idx, set_idx)
        if cache_set[victim_idx] and cache_set[victim_idx].valid:
            if cache_set[victim_idx].dirty:
                # Writeback dirty data to memory
                wb_address = cache_set[victim_idx].tag
                wb_data = cache_set[victim_idx].data
                self.memory.write(wb_address, wb_data, self.block_size)
                self.stats['offchip_writes'] += 1
                self.stats['offchip_bandwidth'] += self.block_size
            
            # Invalidate victim
            cache_set[victim_idx].valid = False
        
        # Issue memory read request
        mem_request_id = self.memory.read(address, self.block_size)
        self.outstanding_requests[mem_request_id] = {
            'type': OperationType.FETCH,
            'address': address
        }
        
        #self.bank_busy[bank_idx] = False  # Release bank
        self.banks_to_release.append(bank_idx)
        return True
    
    def _process_fetch_operation(self, bank_idx: int, operation: dict) -> bool:
        """Process a fetch operation"""
        address = operation['address']
        set_idx = operation['set_idx']
        cache_set = self.banks[bank_idx][set_idx]
        
        # Check if already in cache
        for entry in cache_set:
            if entry and entry.valid and entry.tag == address:
                self._update_priority(entry, OperationType.FETCH)
                self._release_bank(bank_idx)
                return True
        
        # Add to fetch queue if not in cache
        self.fetch_queue.append(address)
        self.stats['fetches'] += 1
        
        # Initiate fetch
        request_id = f"fetch_{address}_{self.stats['fetches']}"
        self.outstanding_requests[request_id] = {
            'type': OperationType.FETCH,
            'address': address,
            'data': None
        }
        self.request_latency[request_id] = self.memory_latency
        
        self._release_bank(bank_idx)
        return True
    
    def consume(self, address: int) -> Tuple[Optional[any], bool]:
        """Consume operation - returns (data, valid)"""
        bank_idx, set_idx = self._get_bank_and_set(address)
        
        # Check for bank conflict
        if self.bank_busy[bank_idx]:
            # Check if this operation is already pending
            for op in self.pending_queue:
                if op['type'] == OperationType.CONSUME and op['address'] == address:
                    return None, False  # Already pending, ignore duplicate
            
            # Bank is busy, add to pending queue
            operation = {
                'type': OperationType.CONSUME,
                'address': address,
                'set_idx': set_idx
            }
            self.pending_queue.append(operation)
            return None, False
        
        # Bank is available, mark it as busy
        self.bank_busy[bank_idx] = True
        
        # Check cache
        cache_set = self.banks[bank_idx][set_idx]
        
        # Check for hit
        for entry in cache_set:
            if entry and entry.valid and entry.tag == address:
                data = entry.data
                entry.valid = False  # Invalidate after consume
                self.stats['consume_hits'] += 1
                self.stats['consumes'] += 1
                self.stats['onchip_bandwidth'] += self.block_size
                #self.bank_busy[bank_idx] = False  # Release bank
                self.banks_to_release.append(bank_idx)
                return data, True
        
        # Cache miss - check if there's already an outstanding request
        for request in self.outstanding_requests.values():
            if request['type'] == OperationType.CONSUME and request['address'] == address:
                #self.bank_busy[bank_idx] = False  # Release bank
                self.banks_to_release.append(bank_idx)
                return None, False  # Already outstanding, ignore duplicate
        
        # Cache miss
        self.stats['consume_misses'] += 1
        self.stats['consumes'] += 1
        self.stats['offchip_reads'] += 1
        self.stats['offchip_bandwidth'] += self.block_size
        
        # Issue memory read request for consume
        mem_request_id = self.memory.read(address, self.block_size)
        self.outstanding_requests[mem_request_id] = {
            'type': OperationType.CONSUME,
            'address': address
        }
        
        #self.bank_busy[bank_idx] = False  # Release bank
        self.banks_to_release.append(bank_idx)
        return None, False
    
    def _process_consume_operation(self, bank_idx: int, operation: dict) -> Tuple[Optional[any], bool]:
        """Process a consume operation"""
        address = operation['address']
        set_idx = operation['set_idx']
        cache_set = self.banks[bank_idx][set_idx]
        
        # Check for hit
        for entry in cache_set:
            if entry and entry.valid and entry.tag == address:
                data = entry.data
                entry.valid = False  # Invalidate
                self.stats['consume_hits'] += 1
                self.stats['consumes'] += 1
                self.stats['onchip_bandwidth'] += self.block_size
                self._release_bank(bank_idx)
                return data, True
        
        # Cache miss
        self.stats['consume_misses'] += 1
        self.stats['consumes'] += 1
        self.stats['offchip_reads'] += 1
        self.stats['offchip_bandwidth'] += self.block_size
        
        request_id = f"consume_{address}_{self.stats['consumes']}"
        self.outstanding_requests[request_id] = {
            'type': OperationType.CONSUME,
            'address': address,
            'data': None
        }
        self.request_latency[request_id] = self.memory_latency
        
        self._release_bank(bank_idx)
        return None, False
    
    def tick(self) -> None:
        """Clock tick - update latencies and handle bank queues"""
        # Tick the HBM memory
        for bank_idx in self.banks_to_release:
            self.bank_busy[bank_idx] = False
        self.banks_to_release = []
        
        completed_memory_requests = self.memory.tick()
        
        print(f"FiberCache tick: {len(completed_memory_requests)} completed memory requests")
        print(f"Outstanding requests: {len(self.outstanding_requests)}")
        print(f"Pending queue: {len(self.pending_queue)}")
        
        # Handle completed memory requests
        for mem_request in completed_memory_requests:
            if mem_request['id'] in self.outstanding_requests:
                cache_request = self.outstanding_requests.pop(mem_request['id'])
                address = cache_request['address']
                data = mem_request['data']
                
                print(f"  Completed request for address {address}, type: {cache_request.get('type', 'unknown')}")
                
                # Insert data into cache
                bank_idx, set_idx = self._get_bank_and_set(address)
                cache_set = self.banks[bank_idx][set_idx]
                
                victim_idx = self._find_victim(bank_idx, set_idx)
                
                # Handle writeback if victim is dirty
                if cache_set[victim_idx] and cache_set[victim_idx].valid and cache_set[victim_idx].dirty:
                    # Writeback dirty data to memory
                    wb_address = cache_set[victim_idx].tag
                    wb_data = cache_set[victim_idx].data
                    self.memory.write(wb_address, wb_data, self.block_size)
                    self.stats['offchip_writes'] += 1
                    self.stats['offchip_bandwidth'] += self.block_size
                
                # Insert new entry
                new_entry = CacheEntry(data, rrpv=2, priority=0)
                new_entry.tag = address
                cache_set[victim_idx] = new_entry
        
        # Process pending queue
        processed_operations = []
        for i, operation in enumerate(list(self.pending_queue)):
            op_type = operation['type']
            address = operation['address']
            bank_idx, set_idx = self._get_bank_and_set(address)
            
            # Check if bank is available
            if not self.bank_busy[bank_idx]:
                # Remove from pending queue
                processed_operations.append(i)
                
                # Process operation based on type
                if op_type == OperationType.READ:
                    # Re-execute read operation
                    self.read(operation['address'])
                elif op_type == OperationType.WRITE:
                    # Re-execute write operation
                    self.write(operation['address'], operation['data'])
                elif op_type == OperationType.FETCH:
                    # Re-execute fetch operation
                    self.fetch(operation['address'])
                elif op_type == OperationType.CONSUME:
                    # Re-execute consume operation
                    self.consume(operation['address'])
        
        # Remove processed operations from pending queue
        for i in sorted(processed_operations, reverse=True):
            self.pending_queue.pop(i)
        
        # Bank queues are no longer needed since we handle bank conflicts directly in operations
    
    def get_stats(self) -> Dict:
        """Return cache statistics"""
        total_accesses = (self.stats['read_hits'] + self.stats['read_misses'] + 
                         self.stats['write_hits'] + self.stats['write_misses'] +
                         self.stats['consume_hits'] + self.stats['consume_misses'])
        
        hit_rate = 0
        if total_accesses > 0:
            hit_rate = ((self.stats['read_hits'] + self.stats['write_hits'] + 
                        self.stats['consume_hits']) / total_accesses) * 100
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'total_accesses': total_accesses
        }
    
    def debug_cache_set(self, bank_idx: int, set_idx: int) -> None:
        """Debug print a cache set to see priority and RRPV values"""
        cache_set = self.banks[bank_idx][set_idx]
        print(f"Cache set [{bank_idx}][{set_idx}]:")
        for i, entry in enumerate(cache_set):
            if entry and entry.valid:
                print(f"  Way {i}: tag={hex(entry.tag) if entry.tag is not None else None}, "
                      f"priority={entry.priority}, RRPV={entry.rrpv}, dirty={entry.dirty}")
            else:
                print(f"  Way {i}: <empty>")