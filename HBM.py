import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

class HBMChannel:
    """Represents a single HBM channel with bandwidth constraints"""
    def __init__(self, channel_id: int, bandwidth_gbps: float = 8.0, bus_width_bits: int = 64):
        self.channel_id = channel_id
        self.bandwidth_gbps = bandwidth_gbps
        
        # Convert bandwidth to bytes per cycle (assuming 1GHz clock)
        self.bytes_per_cycle = (bandwidth_gbps * 1e9) / (8 * 1e9)  # 8 bits per byte, 1GHz clock
        
        # Queue of pending requests - this handles channel contention
        self.request_queue = deque()
        
        # Current transaction being processed
        self.current_transaction = None
        self.remaining_cycles = 0
        
    def add_request(self, request) -> None:
        """Add a request to this channel's queue"""
        self.request_queue.append(request)
        
    def tick(self) -> Optional[Dict]:
        """Process one cycle of this channel"""
        # If currently processing a transaction
        if self.current_transaction:
            # Transfer bytes according to bandwidth
            self.remaining_cycles -= 1
            
            # Check if transaction is complete
            if self.remaining_cycles <= 0:
                completed_request = self.current_transaction
                self.current_transaction = None
                return completed_request
                
        # If no current transaction and queue not empty, start new transaction
        if not self.current_transaction and self.request_queue:
            # Channel contention handling: processes one request at a time from queue
            self.current_transaction = self.request_queue.popleft()
            # Calculate cycles needed based on bandwidth
            bytes_to_transfer = self.current_transaction['size']
            self.remaining_cycles = int(np.ceil(bytes_to_transfer / self.bytes_per_cycle))
            
        return None

class HBMMemory:
    """Simulates HBM memory with multiple channels and bandwidth constraints"""
    def __init__(self, num_channels: int = 16, channel_bandwidth_gbps: float = 12.0, 
                 bus_width_bits: int = 64, latency_cycles: int = 100):
        self.num_channels = num_channels
        self.channel_bandwidth_gbps = channel_bandwidth_gbps
        self.bus_width_bits = bus_width_bits
        self.latency_cycles = latency_cycles
        
        # Total bandwidth in GB/s
        self.total_bandwidth_gbps = num_channels * channel_bandwidth_gbps
        
        # Create HBM channels
        self.channels = []
        for i in range(num_channels):
            self.channels.append(HBMChannel(i, channel_bandwidth_gbps, bus_width_bits))
        
        # Memory storage (simple dictionary)
        self.memory_data = {}
        
        # Outstanding requests tracking
        self.outstanding_requests = {}
        self.request_latencies = {}
        self.next_request_id = 0

        self.completed_request = []
        
        # Statistics
        self.stats = {
            'reads_completed': 0,
            'writes_completed': 0,
            'total_bytes_read': 0,
            'total_bytes_written': 0,
            'total_cycles': 0,
            'channel_utilization': [0] * num_channels
        }
    
    def read(self, address: int, size: int) -> int:
        """Initiate a read request"""
        request_id = self._create_request('read', address, size)
        return request_id
    
    def write(self, address: int, data: Any, size: int) -> int:
        """Initiate a write request"""
        request_id = self._create_request('write', address, size, data)
        return request_id
    
    def _create_request(self, operation: str, address: int, size: int, data: Any = None) -> int:
        """Create a memory request and assign to appropriate channel"""
        request_id = self.next_request_id
        self.next_request_id += 1
        
        # Determine which channel based on address
        channel_id = (address // 64) % self.num_channels  # Simple channel mapping
        
        request = {
            'id': request_id,
            'operation': operation,
            'address': address,
            'size': size,
            'data': data,
            'channel_id': channel_id,
            'start_cycle': self.stats['total_cycles']
        }
        
        # Add request to channel queue
        self.channels[channel_id].add_request(request)
        
        # Track outstanding request (latency will be added when transfer completes)
        self.outstanding_requests[request_id] = request
        
        return request_id
    
    def tick(self) -> List[Dict]:
        """Simulate one memory cycle"""
        self.completed_requests = []
        
        # Update cycle count
        self.stats['total_cycles'] += 1
        print(f"HBM tick: {len(self.outstanding_requests)} outstanding requests")
        for req_id, req in self.outstanding_requests.items():
            status = 'waiting for latency' if req_id in self.request_latencies else 'in channel queue'
            print(f"  Request {req_id}: {req['operation']} at {req['address']}, status: {status}")
        # Process each channel
        for channel_id, channel in enumerate(self.channels):
            # Check if channel is busy (transferring data)
            if channel.current_transaction:
                self.stats['channel_utilization'][channel_id] += 1
            
            # Process the channel
            completed = channel.tick()
            if completed:
                # Transfer is complete, now apply latency
                request_id = completed['id']
                
                # Initialize latency if this is the first time we see this request
                if request_id not in self.request_latencies:
                    self.request_latencies[request_id] = self.latency_cycles
                    # Keep track of request waiting for latency
                    self.outstanding_requests[request_id]['status'] = 'latency'
        
        # Process latencies for requests waiting for latency to complete
        for request_id, latency in list(self.request_latencies.items()):
            self.request_latencies[request_id] -= 1
            
            if self.request_latencies[request_id] <= 0:
                if request_id in self.outstanding_requests:
                    request = self.outstanding_requests[request_id]
                    self._complete_request(request)
                    self.completed_requests.append(request)
                    del self.request_latencies[request_id]
        
        return self.completed_requests
    
    def _complete_request(self, request: Dict) -> None:
        """Complete a memory request"""
        if request['operation'] == 'read':
            # Simulate reading data
            if request['address'] not in self.memory_data:
                request['data'] = 0 #f"data_at_{hex(request['address'])}"
            else:
                request['data'] = self.memory_data[request['address']]
            
            self.stats['reads_completed'] += 1
            self.stats['total_bytes_read'] += request['size']
            
        elif request['operation'] == 'write':
            # Store written data
            self.memory_data[request['address']] = request['data']
            self.stats['writes_completed'] += 1
            self.stats['total_bytes_written'] += request['size']
        
        # Remove from outstanding requests
        if request['id'] in self.outstanding_requests:
            del self.outstanding_requests[request['id']]
    
    def is_request_complete(self, request_id: int) -> bool:
        """Check if a request has completed"""
        return request_id not in self.outstanding_requests
    
    def get_request_data(self, request_id: int) -> Optional[Any]:
        """Get the data from a completed read request"""
        # In a real implementation, we would track completed requests
        # For simplicity, we'll return a dummy value
        return f"data_for_request_{request_id}"
    
    def get_stats(self) -> Dict:
        """Return memory statistics"""
        avg_channel_utilization = sum(self.stats['channel_utilization']) / (
            self.stats['total_cycles'] * self.num_channels) if self.stats['total_cycles'] > 0 else 0
        
        return {
            **self.stats,
            'average_channel_utilization': avg_channel_utilization,
            'effective_bandwidth_gbps': (
                (self.stats['total_bytes_read'] + self.stats['total_bytes_written']) /
                self.stats['total_cycles'] * 1e9 / 1e9
            ) if self.stats['total_cycles'] > 0 else 0
        }
