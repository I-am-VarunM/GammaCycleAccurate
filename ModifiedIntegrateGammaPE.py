class ProcessingElement:
    def __init__(self, pe_id, radix=64, debug=True):
        # Basic configuration
        self.pe_id = pe_id
        self.radix = radix
        self.debug = debug
        
        # Performance tracking
        self.total_cycles = 0
        self.active_cycles = 0
        self.stall_cycles = 0
        self.total_energy = 0.0
        
        # Energy costs
        self.energy_merge = 0.05
        self.energy_multiply = 0.1
        self.energy_add = 0.05
        
        # Operation counters
        self.num_reads = 0
        self.num_writes = 0
        self.num_consumes = 0
        self.num_multiplies = 0
        self.num_adds = 0
        self.num_merges = 0
        
        # PE state
        self.idle = True
        self.stalled = False
        self.pending_callback = None
        
        # Task processing state
        self.current_task = None
        self.task_queue = []
        self.input_fibers = []
        self.scaling_factors = []
        self.scaling_factors_valid = False  # New signal to indicate if scaling factors are valid
        self.fiber_indices = []
        self.accumulator = {}
        
        # CSR-related state
        self.row_loading_state = []  # For tracking the loading of CSR rows
        
        # Pipeline stages
        self.pipeline_started = False
        self.fetch_stage = {"valid": False, "data": None}
        self.merge_stage = {"valid": False, "data": None, "fiber_idx": -1}
        self.compute_stage = {"valid": False, "coord": -1, "value": 0.0, "scale_factor": 0.0}
        
        # Cache line size (in elements)
        self.cache_line_size = 16
        
        # Debug info
        self.debug_log = []
    
    def log(self, message):
        """Add a debug message to the log if debugging is enabled"""
        if self.debug:
            self.debug_log.append(f"PE {self.pe_id} - Cycle {self.total_cycles}: {message}")
            print(f"PE {self.pe_id} - Cycle {self.total_cycles}: {message}")
    
    def set_fiber_cache(self, fiber_cache):
        self.fiber_cache = fiber_cache
    
    def set_scaling_factors(self, scaling_factors):
        """Set the scaling factors for the current task from the scheduler"""
        self.scaling_factors = scaling_factors.copy()
        self.scaling_factors_valid = True
        self.log(f"Received valid scaling factors: {scaling_factors}")
        
        # Check if we can start the pipeline now
        self._check_start_pipeline()
    
    def assign_task(self, task):
        """Assign a task to this PE
        
        The task format should include:
        - input_row_data: List of tuples with:
          * col_indices_addr: Memory address for column indices
          * values_addr: Memory address for values
          * start_idx: Starting index in the arrays
          * end_idx: Ending index in the arrays
        - input_consume_flags: List of booleans indicating whether to consume (True) or read (False) each input
        - output_col_indices_addr: Address to write output column indices
        - output_values_addr: Address to write output values
        """
        # Add task to queue
        self.task_queue.append(task)
        self.log(f"Task assigned with {len(task['input_row_data'])} input fibers")
        
        if self.idle:
            self._start_next_task()
    
    def _start_next_task(self):
        """Start processing the next task"""
        if not self.task_queue:
            self.idle = True
            return
            
        self.idle = False
        self.current_task = self.task_queue.pop(0)
        
        # Initialize processing state
        num_inputs = len(self.current_task["input_row_data"])
        self.input_fibers = [None] * num_inputs
        self.scaling_factors = []  # Will be set by scheduler via set_scaling_factors
        self.scaling_factors_valid = False  # Reset validity
        self.fiber_indices = [0] * num_inputs
        self.accumulator = {}
        
        # Reset pipeline stages
        self.pipeline_started = False
        self.fetch_stage = {"valid": False, "data": None}
        self.merge_stage = {"valid": False, "data": None, "fiber_idx": -1}
        self.compute_stage = {"valid": False, "coord": -1, "value": 0.0, "scale_factor": 0.0}
        
        # Set up row loading state for each input row
        self.row_loading_state = []
        for i, row_data in enumerate(self.current_task["input_row_data"]):
            col_indices_addr = row_data[0]
            values_addr = row_data[1]
            start_idx = row_data[2]
            end_idx = row_data[3]
            should_consume = self.current_task.get("input_consume_flags", [False] * num_inputs)[i]
            
            self.row_loading_state.append({
                'idx': i,
                'col_indices_addr': col_indices_addr,
                'values_addr': values_addr,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'should_consume': should_consume,
                'col_indices': None,
                'values': None,
                'loaded': False,
                'current_block_index': 0,
                'total_blocks': (end_idx - start_idx + self.cache_line_size - 1) // self.cache_line_size,
                'col_indices_collected': [],
                'values_collected': []
            })
        
        self.log(f"Starting task with {num_inputs} input rows")
        
        # Start loading the rows
        self._load_input_rows()
    
    def _check_start_pipeline(self):
        """Check if we can start the pipeline"""
        if not self.pipeline_started:
            all_rows_loaded = all(state['loaded'] for state in self.row_loading_state) if self.row_loading_state else False
            
            if all_rows_loaded and self.scaling_factors_valid:
                self.log("All rows loaded and scaling factors valid, starting pipeline")
                self.pipeline_started = True
            elif all_rows_loaded:
                self.log("All rows loaded but waiting for valid scaling factors")
            elif self.scaling_factors_valid:
                self.log("Scaling factors valid but waiting for all rows to load")
    
    def _load_input_rows(self):
        """Load all the input rows from CSR format"""
        loading_started = False
        for i, state in enumerate(self.row_loading_state):
            if not state['loaded']:
                loading_started = True
                self._load_row(i)
                break  # Load one row at a time to model pipeline behavior
        
        if not loading_started:
            # All rows are loaded, check if we can start the pipeline
            self._check_start_pipeline()
    
    def _load_row(self, row_idx):
        """Load a row that may span multiple cache lines"""
        state = self.row_loading_state[row_idx]
        
        # Calculate current block address
        block_index = state['current_block_index']
        elements_per_block = self.cache_line_size
        
        # Calculate offsets for the current block
        block_offset = block_index * elements_per_block
        col_indices_block_addr = state['col_indices_addr'] + block_offset
        values_block_addr = state['values_addr'] + block_offset
        
        self.log(f"Loading block {block_index+1}/{state['total_blocks']} for row {row_idx}")
        
        # Fetch column indices for this block
        if state['should_consume']:
            col_idx_data, valid = self.fiber_cache.consume(col_indices_block_addr)
            if valid:
                self.num_consumes += 1
        else:
            col_idx_data, valid = self.fiber_cache.read(col_indices_block_addr)
            if valid:
                self.num_reads += 1
        
        if not valid:
            self.log(f"  Column indices block at {col_indices_block_addr} not ready, stalling")
            self._stall(1, lambda: self._load_row(row_idx))
            return
        
        # Fetch values for this block
        if state['should_consume']:
            values_data, valid = self.fiber_cache.consume(values_block_addr)
            if valid:
                self.num_consumes += 1
        else:
            values_data, valid = self.fiber_cache.read(values_block_addr)
            if valid:
                self.num_reads += 1
        
        if not valid:
            self.log(f"  Values block at {values_block_addr} not ready, stalling")
            self._stall(1, lambda: self._load_row(row_idx))
            return
        
        # Determine how many elements we actually need from this block
        elements_remaining = (state['end_idx'] - state['start_idx']) - len(state['col_indices_collected'])
        elements_in_block = min(len(col_idx_data), elements_remaining)
        
        # Collect the data from this block
        state['col_indices_collected'].extend(col_idx_data[:elements_in_block])
        state['values_collected'].extend(values_data[:elements_in_block])
        
        self.log(f"  Collected {elements_in_block} elements from block {block_index+1}")
        
        # Move to next block if needed
        state['current_block_index'] += 1
        if state['current_block_index'] < state['total_blocks'] and len(state['col_indices_collected']) < (state['end_idx'] - state['start_idx']):
            self.log(f"  More blocks needed for row {row_idx}, continuing to next block")
            # Continue with the next block in the next cycle
            return
        
        # All blocks for this row have been fetched
        # Create the complete fiber
        fiber = []
        for j in range(len(state['col_indices_collected'])):
            fiber.append((state['col_indices_collected'][j], state['values_collected'][j]))
        
        self.input_fibers[row_idx] = fiber
        state['loaded'] = True
        
        self.log(f"Row {row_idx} fully loaded: {fiber}")
        
        # Continue loading other rows
        self._load_input_rows()
    
    def _process_pipeline(self):
        """Process the pipeline stages in a single cycle"""
        # Debug current pipeline state
        self.log(f"Pipeline state: fetch={self.fetch_stage['valid']}, merge={self.merge_stage['valid']}, compute={self.compute_stage['valid']}")
        
        # Stage 3: Compute - Process accumulation (if there's data in the compute stage)
        if self.compute_stage["valid"]:
            coord = self.compute_stage["coord"]
            value = self.compute_stage["value"]
            scale_factor = self.compute_stage["scale_factor"]
            
            # Scale the value
            scaled_value = value * scale_factor
            self.num_multiplies += 1
            self.total_energy += self.energy_multiply
            self.log(f"  COMPUTE: Scaling {value} * {scale_factor} = {scaled_value}")
            
            # Accumulate value
            if coord in self.accumulator:
                old_value = self.accumulator[coord]
                self.accumulator[coord] += scaled_value
                self.num_adds += 1
                self.total_energy += self.energy_add
                self.log(f"  COMPUTE: Accumulating to existing value at coord {coord}: {old_value} + {scaled_value} = {self.accumulator[coord]}")
            else:
                self.accumulator[coord] = scaled_value
                self.log(f"  COMPUTE: New accumulator entry at coord {coord} = {scaled_value}")
            
            # Compute stage is now empty
            self.compute_stage["valid"] = False
            self.active_cycles += 1
        
        # Stage 2: Merge - Move data from merge stage to compute stage
        if self.merge_stage["valid"] and not self.compute_stage["valid"]:
            # Transfer data from merge to compute stage
            self.compute_stage["valid"] = True
            self.compute_stage["coord"] = self.merge_stage["data"][0]
            self.compute_stage["value"] = self.merge_stage["data"][1]
            self.compute_stage["scale_factor"] = self.scaling_factors[self.merge_stage["fiber_idx"]]
            
            self.log(f"  MERGE: Moving data to compute stage - Coord={self.compute_stage['coord']}, Value={self.compute_stage['value']}")
            
            # Count merge operation
            self.num_merges += 1
            self.total_energy += self.energy_merge
            
            # Merge stage is now empty
            self.merge_stage["valid"] = False
        
        # Stage 1: Fetch - Find the next element with minimum coordinate
        if self.pipeline_started and not self.merge_stage["valid"]:
            # Check if all inputs are exhausted
            all_exhausted = True
            for i in range(len(self.input_fibers)):
                if self.fiber_indices[i] < len(self.input_fibers[i]):
                    all_exhausted = False
                    break
            
            if all_exhausted:
                self.log(f"FETCH: All inputs exhausted, pipeline draining")
                if not self.compute_stage["valid"] and not self.merge_stage["valid"]:
                    self.log(f"Pipeline empty, merge complete")
                    self.log(f"Final accumulator: {self.accumulator}")
                    self._write_output()
                    return
            else:
                # Find the minimum coordinate across all fibers
                min_coord = float('inf')
                min_idx = -1
                min_data = None
                
                for i in range(len(self.input_fibers)):
                    if self.fiber_indices[i] < len(self.input_fibers[i]):
                        coord = self.input_fibers[i][self.fiber_indices[i]][0]
                        self.log(f"  FETCH: Checking fiber {i}: coord={coord}, index={self.fiber_indices[i]}")
                        if coord < min_coord:
                            min_coord = coord
                            min_idx = i
                            min_data = self.input_fibers[i][self.fiber_indices[i]]
                
                if min_idx >= 0:
                    # Advance the fiber index
                    self.fiber_indices[min_idx] += 1
                    
                    # Move data to merge stage
                    self.merge_stage["valid"] = True
                    self.merge_stage["data"] = min_data
                    self.merge_stage["fiber_idx"] = min_idx
                    
                    self.log(f"  FETCH: Found min coordinate {min_coord} with value {min_data[1]} from fiber {min_idx}")
    
    def _write_output(self):
        """Write the output row in CSR format"""
        # Sort the accumulator by coordinate
        sorted_output = sorted(self.accumulator.items())
        
        # Split into column indices and values
        col_indices = [coord for coord, _ in sorted_output]
        values = [value for _, value in sorted_output]
        
        # Write column indices to output address
        self.log(f"Writing column indices to address {self.current_task['output_col_indices_addr']}")
        success = self.fiber_cache.write(self.current_task['output_col_indices_addr'], col_indices)
        
        if not success:
            # Need to wait
            self.log(f"  Write of column indices requires waiting, stalling")
            self._stall(1, lambda: self._write_output())
            return
        
        self.num_writes += 1
        
        # Write values to output address
        self.log(f"Writing values to address {self.current_task['output_values_addr']}")
        success = self.fiber_cache.write(self.current_task['output_values_addr'], values)
        
        if not success:
            # Need to wait
            self.log(f"  Write of values requires waiting, stalling")
            self._stall(1, lambda: self._write_output())
            return
        
        self.num_writes += 1
        
        # Complete the task
        self.log(f"Output write complete")
        self._complete_task()
    
    def _complete_task(self):
        """Complete the current task"""
        self.log(f"Task completed")
        self.idle = True
        self.scaling_factors_valid = False  # Reset for next task
        
        if self.task_queue:
            self._start_next_task()
    
    def _stall(self, cycles, callback):
        """Stall the PE for the specified number of cycles"""
        self.stalled = True
        self.stall_cycles_remaining = cycles
        self.pending_callback = callback
        self.log(f"PE stalled for {cycles} cycles")
    
    def tick(self):
        """Process one cycle"""
        self.total_cycles += 1
        
        if self.debug:
            self.log(f"--- TICK {self.total_cycles} ---")
            self.log(f"  State: idle={self.idle}, stalled={self.stalled}, pipeline_started={self.pipeline_started}, scaling_factors_valid={self.scaling_factors_valid}")
        
        if self.idle:
            if self.debug:
                self.log(f"  PE is idle")
            return
        
        if self.stalled:
            self.stall_cycles += 1
            self.stall_cycles_remaining -= 1
            if self.debug:
                self.log(f"  Stalled, {self.stall_cycles_remaining} cycles remaining")
            
            if self.stall_cycles_remaining <= 0:
                self.log(f"Stall complete, executing callback")
                self.stalled = False
                callback = self.pending_callback
                self.pending_callback = None
                callback()
        elif self.pipeline_started:
            # If not stalled and pipeline has started, process the pipeline
            self._process_pipeline()
        else:
            # Check if we need to load more rows
            all_loaded = all(state['loaded'] for state in self.row_loading_state) if self.row_loading_state else False
            if not all_loaded:
                self.log(f"Still loading row data")
                self._load_input_rows()
            else:
                print("Load Complete")
                self._check_start_pipeline()
    
    def get_stats(self):
        """Return statistics about PE operation"""
        return {
            "total_cycles": self.total_cycles,
            "active_cycles": self.active_cycles,
            "stall_cycles": self.stall_cycles,
            "utilization": self.active_cycles / max(1, self.total_cycles),
            "total_energy": self.total_energy,
            "num_reads": self.num_reads,
            "num_writes": self.num_writes,
            "num_consumes": self.num_consumes,
            "num_multiplies": self.num_multiplies,
            "num_adds": self.num_adds,
            "num_merges": self.num_merges
        }
