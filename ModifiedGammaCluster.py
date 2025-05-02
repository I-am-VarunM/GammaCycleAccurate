from ModifiedIntegrateGammaPE import ProcessingElement
from FiberCache import FiberCache
class GammaCluster:
    def __init__(self, num_pes=32, radix=64, fiber_cache=None):
        """Initialize a GammaCluster with a scheduler and multiple PEs
        
        Args:
            num_pes: Number of Processing Elements
            radix: Radix of each PE merger (max number of fibers to merge in a single pass)
            fiber_cache: Optional FiberCache instance to use. If None, creates a new one
        """
        # Use provided FiberCache or create new one
        self.fiber_cache = fiber_cache if fiber_cache else FiberCache(
            num_banks=48, bank_size=64*1024, associativity=16, block_size=64)
        
        # Create the PEs with the updated PE class for CSR format
        self.pes = []
        for pe_id in range(num_pes):
            pe = ProcessingElement(pe_id=pe_id, radix=radix, debug=True)
            pe.set_fiber_cache(self.fiber_cache)
            self.pes.append(pe)
        
        # Initialize scheduler state
        self.radix = radix
        self.pending_tasks = []  # Queue of tasks to be processed
        self.total_cycles = 0
        
        # Stats
        self.stats = {
            'total_cycles': 0,
            'pe_utilization': 0,
            'memory_accesses': 0
        }
    
    def schedule_task(self, task):
        """Schedule a CSR-based task for execution
        
        Args:
            task: Task specification for CSR format
                - input_row_data: List of tuples (col_indices_addr, values_addr, start_idx, end_idx)
                - input_consume_flags: List of booleans for consuming inputs
                - scaling_factors: List of scaling factors for inputs
                - output_col_indices_addr: Address for output column indices
                - output_values_addr: Address for output values
        """
        print(f"Scheduling task with {len(task['input_row_data'])} input rows")
        print(f"  Output addresses: col_indices={task['output_col_indices_addr']}, values={task['output_values_addr']}")
        
        # Check if task needs to be decomposed (if number of inputs exceeds PE radix)
        if len(task['input_row_data']) <= self.radix:
            # Simple case: assign directly to a PE
            self.pending_tasks.append(task)
        else:
            # Complex case: break into multiple tasks
            print(f"  Task has {len(task['input_row_data'])} inputs, decomposing")
            subtasks = self._decompose_task(task)
            self.pending_tasks.extend(subtasks)
    
    def _decompose_task(self, task):
        """Break a large task into smaller ones that fit within PE radix
        
        Args:
            task: Original task with input_row_data exceeding radix
            
        Returns:
            List of subtasks that collectively perform the original task
        """
        # Temporary addresses for partial outputs
        next_temp_col_indices_addr = 20000  # Start temp addresses at 20000
        next_temp_values_addr = 30000
        
        # Gather input data
        inputs = task['input_row_data']
        scaling_factors = task['scaling_factors']
        
        # Group inputs into chunks of radix size
        chunks = []
        for i in range(0, len(inputs), self.radix):
            chunk = inputs[i:i+self.radix]
            chunks.append(chunk)
        
        print(f"  Created {len(chunks)} chunks")
        
        # Create leaf tasks
        subtasks = []
        partial_outputs = []
        
        for i, chunk in enumerate(chunks):
            # Assign temp addresses for this partial output
            temp_col_indices_addr = next_temp_col_indices_addr
            temp_values_addr = next_temp_values_addr
            next_temp_col_indices_addr += self.radix
            next_temp_values_addr += self.radix
            
            # Create subtask for this chunk
            chunk_scaling_factors = scaling_factors[i*self.radix:(i+1)*self.radix]
            chunk_consume_flags = [False] * len(chunk)  # Don't consume inputs
            
            subtask = {
                "input_row_data": chunk,
                "input_consume_flags": chunk_consume_flags,
                "scaling_factors": chunk_scaling_factors,
                "output_col_indices_addr": temp_col_indices_addr,
                "output_values_addr": temp_values_addr
            }
            
            subtasks.append(subtask)
            partial_outputs.append((temp_col_indices_addr, temp_values_addr))
        
        # Create merge tasks for partial outputs if needed
        while len(partial_outputs) > 1:
            new_partial_outputs = []
            
            for i in range(0, len(partial_outputs), self.radix):
                chunk = partial_outputs[i:i+self.radix]
                
                # If this is the final merge and it's the only one, output to original destination
                if i + self.radix >= len(partial_outputs) and len(new_partial_outputs) == 0:
                    out_col_indices_addr = task['output_col_indices_addr']
                    out_values_addr = task['output_values_addr']
                else:
                    # Otherwise, create new temp addresses
                    out_col_indices_addr = next_temp_col_indices_addr
                    out_values_addr = next_temp_values_addr
                    next_temp_col_indices_addr += self.radix
                    next_temp_values_addr += self.radix
                
                # Create the merge task
                input_row_data = []
                for col_addr, val_addr in chunk:
                    # For partial outputs, we use the whole array (no start/end indices)
                    input_row_data.append((col_addr, val_addr, 0, self.radix))
                
                merge_task = {
                    "input_row_data": input_row_data,
                    "input_consume_flags": [True] * len(chunk),  # Consume partial outputs
                    "scaling_factors": [1.0] * len(chunk),       # No scaling for merges
                    "output_col_indices_addr": out_col_indices_addr,
                    "output_values_addr": out_values_addr
                }
                
                subtasks.append(merge_task)
                new_partial_outputs.append((out_col_indices_addr, out_values_addr))
            
            partial_outputs = new_partial_outputs
        
        print(f"  Created {len(subtasks)} subtasks")
        return subtasks
    
    def tick(self):
        """Process one cycle of the GammaCluster
        
        Returns:
            True if there is still work to do, False if all work is complete
        """
        self.total_cycles += 1
        
        # Tick the FiberCache
        self.fiber_cache.tick()
        
        # Assign tasks to idle PEs
        self._assign_tasks()
        
        # Tick all PEs
        for pe in self.pes:
            #print("Happening")
            pe.tick()
        
        # Check if all work is complete
        if not self.pending_tasks and all(pe.idle for pe in self.pes):
            return False  # All work is complete
        
        return True  # More work to do
    
    def _assign_tasks(self):
        """Assign pending tasks to idle PEs"""
        # Find idle PEs
        idle_pes = [pe for pe in self.pes if pe.idle]
        
        # Assign pending tasks to idle PEs
        while idle_pes and self.pending_tasks:
            pe = idle_pes.pop(0)
            task = self.pending_tasks.pop(0)
            pe.assign_task(task)
    
    def run_until_complete(self, max_cycles=10000):
        """Run the GammaCluster until all work is complete or max_cycles is reached"""
        print("\n=== Starting GammaCluster execution ===")
        print(f"Pending tasks: {len(self.pending_tasks)}")
        print(f"PE states: {['idle' if pe.idle else 'busy' for pe in self.pes]}")
        
        for cycle in range(max_cycles):
            print(f"\n----- CYCLE {cycle+1} -----")
            if not self.tick():
                print(f"Completed in {cycle+1} cycles")
                print(f"Final PE states: {['idle' if pe.idle else 'busy' for pe in self.pes]}")
                self.stats['total_cycles'] = cycle+1
                return cycle+1
                
            # Print state after each cycle
            if cycle < 10 or cycle % 10 == 0:  # Print first 10 cycles, then every 10th
                active_pes = sum(1 for pe in self.pes if not pe.idle)
                print(f"  Active PEs: {active_pes}/{len(self.pes)}")
                print(f"  Pending tasks: {len(self.pending_tasks)}")
                print(f"  FiberCache stats: {self.fiber_cache.stats['read_hits']} hits, {self.fiber_cache.stats['read_misses']} misses")
        
        print(f"Warning: Reached maximum cycle count ({max_cycles})")
        self.stats['total_cycles'] = max_cycles
        return max_cycles
    
    def get_stats(self):
        """Get statistics about the GammaCluster's operation"""
        # Collect stats from all PEs
        pe_stats = [pe.get_stats() for pe in self.pes]
        
        # Compute average utilization
        avg_utilization = sum(stats["utilization"] for stats in pe_stats) / len(self.pes)
        
        # Update overall stats
        self.stats['pe_utilization'] = avg_utilization
        self.stats['memory_accesses'] = sum(stats["num_reads"] + stats["num_writes"] + stats["num_consumes"] 
                                           for stats in pe_stats)
        
        return {
            **self.stats,
            'pe_stats': pe_stats
        }