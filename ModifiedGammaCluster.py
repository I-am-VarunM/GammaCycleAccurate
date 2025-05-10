class GammaScheduler:
    def __init__(self, hbm_memory, fiber_cache, num_pes=32, pe_radix=64, debug=True):
        """Initialize the Gamma Scheduler
        
        Args:
            hbm_memory: HBM memory instance
            fiber_cache: FiberCache instance
            num_pes: Number of processing elements available
            pe_radix: Radix of each PE (max number of inputs it can merge)
            debug: Enable debug output
        """
        self.hbm_memory = hbm_memory
        self.fiber_cache = fiber_cache
        self.num_pes = num_pes
        self.pe_radix = pe_radix
        self.debug = debug
        self.elements_per_block = 16  # Number of non-zero elements per block
        
        # References to PE instances (to be set later)
        self.pes = []
        
        # Current matrix being processed
        self.matrix_a_base_addr = None
        self.matrix_a_rows = None
        self.matrix_b_base_addr = None
        self.matrix_b_cols = None
        self.output_matrix_base_addr = None
        
        # Task tracking
        self.next_task_id = 0
        self.next_row_to_schedule = 0
        self.rows_completed = 0
        self.total_rows = 0
        
        # Task scoreboard - main data structure for tracking tasks
        self.scoreboard = {}  # task_id -> task_info
        self.pending_tasks = []  # Tasks ready to be dispatched
        self.running_tasks = {}  # pe_id -> task_id
        self.completed_tasks = []  # List of completed task IDs
        
        # CSR row pointers - needed to determine row sizes
        self.rowptr_a = None
        self.rowptr_a_addr = None
        self.rowptr_a_loaded = False
        
        # Task tree tracking
        self.task_tree = {}  # row_id -> list of task nodes
        
        # Partial output tracking
        self.partial_outputs = {}  # (row_id, level, node_idx) -> addr
        self.next_partial_output_addr = None
        
        # Memory request tracking
        self.outstanding_requests = {}  # request_id -> info
        
        # PE state tracking
        self.pe_ready = [True] * num_pes  # Whether each PE is ready for a new task
        
        # Statistics
        self.cycles = 0
        self.stats = {
            'tasks_created': 0,
            'tasks_dispatched': 0,
            'tasks_completed': 0,
            'memory_requests': 0,
            'idle_cycles': 0,
            'active_cycles': 0,
            'a_rows_loaded': 0,
            'b_rows_loaded': 0,
        }
        
        # Debug log
        self.debug_log = []
    
    def log(self, message):
        """Add a message to the debug log"""
        if self.debug:
            self.debug_log.append(f"Cycle {self.cycles}: {message}")
            print(f"Scheduler - Cycle {self.cycles}: {message}")
    
    def set_pes(self, pes):
        """Set the list of PE instances
        
        Args:
            pes: List of ProcessingElement instances
        """
        self.pes = pes
        if len(pes) != self.num_pes:
            self.log(f"Warning: Expected {self.num_pes} PEs, got {len(pes)}")
            self.num_pes = len(pes)
            self.pe_ready = [True] * self.num_pes
    
    def set_matrix_operation(self, matrix_a_base_addr, matrix_a_rows, 
                            matrix_b_base_addr, matrix_b_cols,
                            output_matrix_base_addr, rowptr_a_addr):
        """Set the matrix operation to be performed
        
        Args:
            matrix_a_base_addr: Base address of matrix A
            matrix_a_rows: Number of rows in matrix A
            matrix_b_base_addr: Base address of matrix B
            matrix_b_cols: Number of columns in matrix B
            output_matrix_base_addr: Base address for the output matrix
            rowptr_a_addr: Address of the row pointers for matrix A
        """
        self.matrix_a_base_addr = matrix_a_base_addr
        self.matrix_a_rows = matrix_a_rows
        self.matrix_b_base_addr = matrix_b_base_addr
        self.matrix_b_cols = matrix_b_cols
        self.output_matrix_base_addr = output_matrix_base_addr
        self.rowptr_a_addr = rowptr_a_addr
        
        # Reset scheduling state
        self.next_row_to_schedule = 0
        self.rows_completed = 0
        self.total_rows = matrix_a_rows
        self.next_task_id = 0
        self.scoreboard = {}
        self.pending_tasks = []
        self.running_tasks = {}
        self.completed_tasks = []
        self.task_tree = {}
        self.partial_outputs = {}
        self.next_partial_output_addr = output_matrix_base_addr + (matrix_a_rows * 2 * 4)  # Assuming 4 bytes per element
        
        # Request row pointers for matrix A in blocks
        # Calculate number of blocks needed for row pointers
        self.rowptr_a_loaded = False
        self.rowptr_a = []
        self.rowptr_blocks_needed = ((matrix_a_rows + 1) + self.elements_per_block - 1) // self.elements_per_block
        self.rowptr_blocks_received = 0
        
        # Request each block of row pointers
        for block_idx in range(self.rowptr_blocks_needed):
            start_element = block_idx * self.elements_per_block
            elements_in_block = min(self.elements_per_block, matrix_a_rows + 1 - start_element)
            block_addr = rowptr_a_addr + (start_element * 4)  # Assuming 4 bytes per element
            
            request_id = self.hbm_memory.read(block_addr, elements_in_block * 4)
            
            self.outstanding_requests[request_id] = {
                'type': 'rowptr_block',
                'block_idx': block_idx,
                'addr': block_addr,
                'size': elements_in_block * 4
            }
            
            self.log(f"Requested row pointer block {block_idx+1}/{self.rowptr_blocks_needed} from address {block_addr}")
            self.stats['memory_requests'] += 1
    
    def _create_task_tree(self, row_id, nonzeros):
        """Create a task tree for a row with more nonzeros than the PE radix
        
        Args:
            row_id: The row ID in matrix A
            nonzeros: The number of nonzeros in the row
            
        Returns:
            A list of tasks organized in a tree structure
        """
        self.log(f"Creating task tree for row {row_id} with {nonzeros} nonzeros")
        
        # Calculate the number of leaf tasks needed
        num_leaf_tasks = (nonzeros + self.pe_radix - 1) // self.pe_radix
        
        # Calculate tree height
        height = 1
        nodes_at_level = num_leaf_tasks
        while nodes_at_level > 1:
            nodes_at_level = (nodes_at_level + self.pe_radix - 1) // self.pe_radix
            height += 1
        
        self.log(f"Task tree height: {height}, leaf tasks: {num_leaf_tasks}")
        
        # Initialize task tree structure
        tree = [[] for _ in range(height)]
        
        # Create leaf tasks (level 0)
        for leaf_idx in range(num_leaf_tasks):
            start_nonzero = leaf_idx * self.pe_radix
            end_nonzero = min((leaf_idx + 1) * self.pe_radix, nonzeros)
            
            leaf_task = {
                'level': 0,
                'node_idx': leaf_idx,
                'start_nonzero': start_nonzero,
                'end_nonzero': end_nonzero,
                'input_tasks': [],  # Leaf tasks have no input tasks
                'output_task': None,  # Will be filled in later
                'task_id': None,  # Will be assigned when scheduled
                'completed': False
            }
            
            tree[0].append(leaf_task)
        
        # Create internal nodes
        for level in range(1, height):
            prev_level_nodes = len(tree[level-1])
            nodes_this_level = (prev_level_nodes + self.pe_radix - 1) // self.pe_radix
            
            for node_idx in range(nodes_this_level):
                start_child = node_idx * self.pe_radix
                end_child = min((node_idx + 1) * self.pe_radix, prev_level_nodes)
                
                # List of child tasks that feed into this node
                input_tasks = tree[level-1][start_child:end_child]
                
                # Create the node
                node_task = {
                    'level': level,
                    'node_idx': node_idx,
                    'input_tasks': input_tasks,
                    'output_task': None,  # Will be filled in for non-root nodes
                    'task_id': None,  # Will be assigned when scheduled
                    'completed': False
                }
                
                # Set the output task of all child nodes
                for child in input_tasks:
                    child['output_task'] = node_task
                
                tree[level].append(node_task)
        
        # Store the task tree
        self.task_tree[row_id] = tree
        
        return tree
    
    def _can_schedule_task(self, task):
        """Check if a task can be scheduled based on its dependencies
        
        Args:
            task: The task to check
            
        Returns:
            bool: True if the task can be scheduled, False otherwise
        """
        # Leaf tasks can always be scheduled
        if task['level'] == 0:
            return True
        
        # For non-leaf tasks, all input tasks must be completed
        return all(input_task['completed'] for input_task in task['input_tasks'])
    
    def _find_ready_task(self):
        """Find a task that is ready to be scheduled
        
        Returns:
            The highest priority task that is ready, or None if no tasks are ready
        """
        # First, check for leaf tasks for new rows
        if self.next_row_to_schedule < self.total_rows and self.rowptr_a_loaded:
            row_id = self.next_row_to_schedule
            row_nnz = self.rowptr_a[row_id + 1] - self.rowptr_a[row_id]
            
            # If this row fits in a single PE, create a simple task
            if row_nnz <= self.pe_radix:
                # Create a simple task for this row
                task = self._create_simple_task(row_id, row_nnz)
                self.pending_tasks.append(task)
                self.next_row_to_schedule += 1
                return task
            else:
                # Create a task tree for this row if not already created
                if row_id not in self.task_tree:
                    self._create_task_tree(row_id, row_nnz)
                
                # Find leaf tasks that are ready
                for leaf_task in self.task_tree[row_id][0]:
                    if leaf_task['task_id'] is None:  # Not yet scheduled
                        task = self._create_task_from_node(row_id, leaf_task)
                        self.pending_tasks.append(task)
                        return task
                
                # If all leaf tasks are scheduled, check for internal nodes
                for level in range(1, len(self.task_tree[row_id])):
                    for node in self.task_tree[row_id][level]:
                        if node['task_id'] is None and self._can_schedule_task(node):
                            task = self._create_task_from_node(row_id, node)
                            self.pending_tasks.append(task)
                            return task
                
                # If all tasks for this row are scheduled, move to next row
                all_scheduled = True
                for level in self.task_tree[row_id]:
                    for node in level:
                        if node['task_id'] is None:
                            all_scheduled = False
                            break
                    if not all_scheduled:
                        break
                
                if all_scheduled:
                    self.next_row_to_schedule += 1
                
                # If no new tasks for this row, check existing pending tasks
                if self.pending_tasks:
                    return self.pending_tasks[0]
                
                # Try scheduling from the next row
                return self._find_ready_task()
        
        # If we have pending tasks, return the highest priority one
        if self.pending_tasks:
            # Priority: higher level tasks first (to reduce partial output footprint)
            highest_priority = None
            for task in self.pending_tasks:
                if 'node' in task and task['node']['level'] > 0:
                    if highest_priority is None or task['node']['level'] > highest_priority['node']['level']:
                        highest_priority = task
            
            # If no high-priority tasks, return the first one
            return highest_priority if highest_priority else self.pending_tasks[0]
        
        return None
    
    def _request_a_row(self, task_id, row_addr, row_nnz):
        """Request an A row from memory, handling block-based access
        
        Args:
            task_id: ID of the task this row is for
            row_addr: Starting address of the row
            row_nnz: Number of nonzeros in the row
        """
        task = self.scoreboard[task_id]
        
        # Calculate number of blocks needed
        num_blocks = (row_nnz + self.elements_per_block - 1) // self.elements_per_block
        
        # Initialize tracking for this request
        task['a_row_blocks_needed'] = num_blocks
        task['a_row_blocks_received'] = 0
        task['a_row_indices'] = []
        task['a_row_values'] = []
        task['a_row_block_requests'] = []
        
        # Request each block
        for block_idx in range(num_blocks):
            start_element = block_idx * self.elements_per_block
            elements_in_block = min(self.elements_per_block, row_nnz - start_element)
            block_addr = row_addr + (start_element * 4)  # Assuming 4 bytes per element
            
            # Request this block
            request_id = self.hbm_memory.read(block_addr, elements_in_block * 4)
            task['a_row_block_requests'].append(request_id)
            
            # Track in outstanding requests
            self.outstanding_requests[request_id] = {
                'type': 'a_row_block',
                'task_id': task_id,
                'block_idx': block_idx,
                'addr': block_addr,
                'size': elements_in_block * 4
            }
            
            self.log(f"Requested A row block {block_idx+1}/{num_blocks} for task {task_id} from address {block_addr}")
            self.stats['memory_requests'] += 1
    
    def _create_simple_task(self, row_id, nnz):
        """Create a simple task for a row that fits in a single PE
        
        Args:
            row_id: The row ID in matrix A
            nnz: The number of nonzeros in the row
            
        Returns:
            A task dictionary ready to be assigned to a PE
        """
        task_id = self.next_task_id
        self.next_task_id += 1
        self.stats['tasks_created'] += 1
        
        # Calculate addresses for this row
        a_row_addr = self.matrix_a_base_addr + self.rowptr_a[row_id] * 4  # Assuming 4 bytes per element
        output_row_addr = self.output_matrix_base_addr + row_id * 2 * 4  # Assuming 4 bytes per element
        
        # Create the task
        task = {
            'id': task_id,
            'type': 'simple',
            'row_id': row_id,
            'input_row_data': [],  # Will be filled with B rows
            'scaling_factors': [],  # Will be filled with A values
            'output_col_indices_addr': output_row_addr,
            'output_values_addr': output_row_addr + self.matrix_b_cols * 4,  # Assuming 4 bytes per element
            'a_row_addr': a_row_addr,
            'a_row_nnz': nnz,
            'a_row_loaded': False,
            'ready': False  # Will be set to True when A row is loaded
        }
        
        # Add to scoreboard
        self.scoreboard[task_id] = task
        
        # Request A row in blocks
        self._request_a_row(task_id, a_row_addr, nnz)
        
        return task
    
    def _create_task_from_node(self, row_id, node):
        """Create a task from a node in the task tree
        
        Args:
            row_id: The row ID in matrix A
            node: The node in the task tree
            
        Returns:
            A task dictionary ready to be assigned to a PE
        """
        task_id = self.next_task_id
        self.next_task_id += 1
        self.stats['tasks_created'] += 1
        node['task_id'] = task_id
        
        # Determine if this is a root task
        is_root = (node['level'] == len(self.task_tree[row_id]) - 1)
        
        # Calculate output address
        output_addr = None
        if is_root:
            # Root node outputs to the final result matrix
            output_addr = self.output_matrix_base_addr + row_id * 2 * 4  # Assuming 4 bytes per element
        else:
            # Internal node outputs to a temporary location
            output_key = (row_id, node['level'], node['node_idx'])
            if output_key not in self.partial_outputs:
                self.partial_outputs[output_key] = self.next_partial_output_addr
                self.next_partial_output_addr += self.matrix_b_cols * 4 * 2  # Space for indices and values
            
            output_addr = self.partial_outputs[output_key]
        
        # Create the task
        task = {
            'id': task_id,
            'type': 'tree_node',
            'row_id': row_id,
            'node': node,
            'input_row_data': [],  # Will be filled based on level
            'scaling_factors': [],  # Will be filled when dispatched
            'output_col_indices_addr': output_addr,
            'output_values_addr': output_addr + self.matrix_b_cols * 4,  # Assuming 4 bytes per element
            'ready': False  # Will be set to True when inputs are ready
        }
        
        # For leaf nodes, we need A row values
        if node['level'] == 0:
            # Calculate A row segment
            a_start = self.rowptr_a[row_id] + node['start_nonzero']
            a_end = self.rowptr_a[row_id] + node['end_nonzero']
            a_row_addr = self.matrix_a_base_addr + a_start * 4  # Assuming 4 bytes per element
            a_row_nnz = a_end - a_start
            
            task['a_row_addr'] = a_row_addr
            task['a_row_nnz'] = a_row_nnz
            task['a_row_loaded'] = False
            
            # Request A row segment in blocks
            self._request_a_row(task_id, a_row_addr, a_row_nnz)
        
        # For internal nodes, inputs come from child tasks
        else:
            task['input_tasks'] = []
            for input_task in node['input_tasks']:
                input_task_id = input_task['task_id']
                if input_task_id is not None:
                    task['input_tasks'].append(input_task_id)
            
            # Check if all input tasks are completed
            if all(task_id in self.completed_tasks for task_id in task['input_tasks']):
                task['ready'] = True
        
        # Add to scoreboard
        self.scoreboard[task_id] = task
        
        return task
    
    def _dispatch_task_to_pe(self, pe_id, task):
        """Dispatch a task to a PE
        
        Args:
            pe_id: The ID of the PE to dispatch to
            task: The task to dispatch
        """
        # Mark PE as busy
        self.pe_ready[pe_id] = False
        
        # Mark task as running
        task['status'] = 'running'
        task['pe_id'] = pe_id
        self.running_tasks[pe_id] = task['id']
        
        # Remove from pending tasks
        if task in self.pending_tasks:
            self.pending_tasks.remove(task)
        
        # Prepare input data
        if task['type'] == 'simple' and task['a_row_loaded']:
            # A simple task needs B rows corresponding to A row nonzeros
            a_row_nnz = task['a_row_nnz']
            scaling_factors = task['a_row_values']
            input_row_data = []
            
            # For each nonzero in A, we need the corresponding B row
            for i in range(a_row_nnz):
                col_idx = task['a_row_indices'][i]
                b_row_addr = self.matrix_b_base_addr + self.rowptr_a[col_idx] * 4  # Assuming 4 bytes per element
                b_row_size = self.rowptr_a[col_idx + 1] - self.rowptr_a[col_idx]
                
                input_row_data.append((b_row_addr, b_row_addr + b_row_size * 4, 0, b_row_size))
            
            # Create the PE task
            pe_task = {
                'input_row_data': input_row_data,
                'scaling_factors': scaling_factors,
                'output_col_indices_addr': task['output_col_indices_addr'],
                'output_values_addr': task['output_values_addr']
            }
            
            # Assign task to PE
            self.pes[pe_id].assign_task(pe_task)
            
            # Send scaling factors to PE
            self.pes[pe_id].set_scaling_factors(scaling_factors)
            
            self.log(f"Dispatched simple task {task['id']} for row {task['row_id']} to PE {pe_id}")
            self.stats['tasks_dispatched'] += 1
        
        elif task['type'] == 'tree_node' and task['ready']:
            if task['node']['level'] == 0:
                # Leaf node in task tree - needs B rows corresponding to A segment
                a_row_nnz = task['a_row_nnz']
                scaling_factors = task['a_row_values']
                input_row_data = []
                
                # For each nonzero in the A segment, we need the corresponding B row
                for i in range(a_row_nnz):
                    col_idx = task['a_row_indices'][i]
                    b_row_addr = self.matrix_b_base_addr + self.rowptr_a[col_idx] * 4  # Assuming 4 bytes per element
                    b_row_size = self.rowptr_a[col_idx + 1] - self.rowptr_a[col_idx]
                    
                    input_row_data.append((b_row_addr, b_row_addr + b_row_size * 4, 0, b_row_size))
                
                # Create the PE task
                pe_task = {
                    'input_row_data': input_row_data,
                    'scaling_factors': scaling_factors,
                    'output_col_indices_addr': task['output_col_indices_addr'],
                    'output_values_addr': task['output_values_addr']
                }
                
                # Assign task to PE
                self.pes[pe_id].assign_task(pe_task)
                
                # Send scaling factors to PE
                self.pes[pe_id].set_scaling_factors(scaling_factors)
                
                self.log(f"Dispatched leaf tree node task {task['id']} for row {task['row_id']} to PE {pe_id}")
                self.stats['tasks_dispatched'] += 1
            
            else:
                # Internal node in task tree - needs partial results from child tasks
                input_row_data = []
                scaling_factors = []
                input_consume_flags = []  # Add this to track which inputs to consume
                
                # For each input task, add its output as an input
                for i, input_task_id in enumerate(task['input_tasks']):
                    input_task = self.scoreboard[input_task_id]
                    input_addr = input_task['output_col_indices_addr']
                    
                    # We don't know the size of the partial output, so use a large value
                    # In a real implementation, we would track sizes of partial outputs
                    input_size = self.matrix_b_cols  # Estimate
                    
                    input_row_data.append((input_addr, input_addr + input_size * 4, 0, input_size))
                    scaling_factors.append(1.0)  # Scale factor is 1.0 for partial outputs
                    input_consume_flags.append(True)  # Mark partial results for consumption
                
                # Create the PE task with consume flags
                pe_task = {
                    'input_row_data': input_row_data,
                    'input_consume_flags': input_consume_flags,
                    'scaling_factors': scaling_factors,
                    'output_col_indices_addr': task['output_col_indices_addr'],
                    'output_values_addr': task['output_values_addr']
                }
                
                # Assign task to PE
                self.pes[pe_id].assign_task(pe_task)
                
                # Send scaling factors to PE
                self.pes[pe_id].set_scaling_factors(scaling_factors)
                
                self.log(f"Dispatched internal tree node task {task['id']} for row {task['row_id']} to PE {pe_id}")
                self.stats['tasks_dispatched'] += 1
        
        else:
            self.log(f"Warning: Tried to dispatch task {task['id']} but it's not ready")
            # Put PE back in ready state
            self.pe_ready[pe_id] = True
            # Remove from running tasks
            if pe_id in self.running_tasks:
                del self.running_tasks[pe_id]
    
    def _handle_task_completion(self, pe_id):
        """Handle completion of a task on a PE
        
        Args:
            pe_id: The ID of the PE that completed a task
        """
        if pe_id in self.running_tasks:
            task_id = self.running_tasks[pe_id]
            task = self.scoreboard[task_id]
            
            # Mark task as completed
            task['status'] = 'completed'
            self.completed_tasks.append(task_id)
            
            self.log(f"Task {task_id} completed on PE {pe_id}")
            self.stats['tasks_completed'] += 1
            
            # If this is a tree node, mark the node as completed
            if task['type'] == 'tree_node':
                task['node']['completed'] = True
                
                # Check if this completes the row
                row_id = task['row_id']
                if task['node']['level'] == len(self.task_tree[row_id]) - 1:
                    self.rows_completed += 1
                    self.log(f"Row {row_id} completed, {self.rows_completed}/{self.total_rows} rows done")
            
            elif task['type'] == 'simple':
                # Simple task completes the row directly
                self.rows_completed += 1
                self.log(f"Row {task['row_id']} completed, {self.rows_completed}/{self.total_rows} rows done")
            
            # Remove from running tasks
            del self.running_tasks[pe_id]
            
            # Mark PE as ready for new tasks
            self.pe_ready[pe_id] = True
    
    def _update_ready_tasks(self):
        """Update the ready status of tasks based on completed dependencies"""
        for task_id, task in self.scoreboard.items():
            # Skip tasks that are already ready, running, or completed
            if task.get('ready', False) or task.get('status') in ['running', 'completed']:
                continue
            
            if task['type'] == 'simple':
                # Simple task is ready when A row is loaded
                if task.get('a_row_loaded', False):
                    task['ready'] = True
                    if task not in self.pending_tasks:
                        self.pending_tasks.append(task)
            
            elif task['type'] == 'tree_node':
                if task['node']['level'] == 0:
                    # Leaf node is ready when A row segment is loaded
                    if task.get('a_row_loaded', False):
                        task['ready'] = True
                        if task not in self.pending_tasks:
                            self.pending_tasks.append(task)
                
                else:
                    # Internal node is ready when all input tasks are completed
                    all_inputs_ready = True
                    for input_task_id in task['input_tasks']:
                        if input_task_id not in self.completed_tasks:
                            all_inputs_ready = False
                            break
                    
                    if all_inputs_ready:
                        task['ready'] = True
                        if task not in self.pending_tasks:
                            self.pending_tasks.append(task)
    
    def _handle_memory_response(self, request):
        """Handle a memory response
        
        Args:
            request: The memory request that completed
        """
        request_id = request['id']
        
        if request_id in self.outstanding_requests:
            req_info = self.outstanding_requests[request_id]
            
            if req_info['type'] == 'rowptr_block':
                # Block of row pointers for matrix A
                block_idx = req_info['block_idx']
                
                # Parse the data
                self.rowptr_a.extend(request['data'])
                self.rowptr_blocks_received += 1
                
                # Check if all blocks received
                if self.rowptr_blocks_received == self.rowptr_blocks_needed:
                    self.rowptr_a_loaded = True
                    self.log(f"Loaded all row pointers for matrix A: {len(self.rowptr_a)} pointers")
            
            elif req_info['type'] == 'a_row_block':
                # A row block
                task_id = req_info['task_id']
                block_idx = req_info['block_idx']
                
                if task_id in self.scoreboard:
                    task = self.scoreboard[task_id]
                    
                    # Parse the A row data into indices and values
                    # In a real implementation, we would need to handle the format correctly
                    # Assuming alternating indices and values in the data
                    data = request['data']
                    indices = []
                    values = []
                    for i in range(0, len(data), 2):
                        if i + 1 < len(data):  # Make sure we have both index and value
                            indices.append(data[i])
                            values.append(data[i+1])
                    
                    # Add this block's data to the task
                    task['a_row_indices'].extend(indices)
                    task['a_row_values'].extend(values)
                    task['a_row_blocks_received'] += 1
                    
                    # Check if all blocks received
                    if task['a_row_blocks_received'] == task['a_row_blocks_needed']:
                        task['a_row_loaded'] = True
                        
                        # Update task readiness
                        if not task.get('ready', False):
                            task['ready'] = True
                            if task not in self.pending_tasks:
                                self.pending_tasks.append(task)
                        
                        self.log(f"Completed loading all blocks for A row in task {task_id}")
                        self.stats['a_rows_loaded'] += 1
            
            # Remove from outstanding requests
            del self.outstanding_requests[request_id]
    
    def tick(self):
        """Process one cycle of the scheduler"""
        self.cycles += 1
        
        # Check for completed memory requests from HBM
        if hasattr(self.hbm_memory, 'completed_requests'):
            for request in self.hbm_memory.completed_requests:
                self._handle_memory_response(request)
        
        # Check for completed tasks on PEs
        for pe_id in range(self.num_pes):
            if pe_id in self.running_tasks and self.pes[pe_id].idle:
                self._handle_task_completion(pe_id)
        
        # Update task readiness based on completed dependencies
        self._update_ready_tasks()
        
        # Dispatch ready tasks to available PEs
        for pe_id in range(self.num_pes):
            if self.pe_ready[pe_id]:
                # Find a ready task to dispatch
                task = self._find_ready_task()
                if task:
                    self._dispatch_task_to_pe(pe_id, task)
                else:
                    # No ready tasks, PE remains idle
                    self.stats['idle_cycles'] += 1
            else:
                # PE is busy
                self.stats['active_cycles'] += 1
        
        # Check if all rows are completed
        if self.rows_completed == self.total_rows and self.total_rows > 0:
            self.log(f"All {self.rows_completed} rows completed!")
            return True  # Signal completion of current matrix operation
        
        return False  # Not done yet
    
    def is_complete(self):
        """Check if the current matrix operation is complete"""
        return self.rows_completed == self.total_rows and self.total_rows > 0
    
    def get_stats(self):
        """Get statistics about the scheduler's operation"""
        return {
            'cycles': self.cycles,
            'rows_completed': self.rows_completed,
            'total_rows': self.total_rows,
            'tasks_created': self.stats['tasks_created'],
            'tasks_dispatched': self.stats['tasks_dispatched'],
            'tasks_completed': self.stats['tasks_completed'],
            'memory_requests': self.stats['memory_requests'],
            'idle_cycles': self.stats['idle_cycles'],
            'active_cycles': self.stats['active_cycles'],
            'utilization': self.stats['active_cycles'] / max(1, self.cycles),
            'a_rows_loaded': self.stats['a_rows_loaded'],
            'b_rows_loaded': self.stats['b_rows_loaded'],
        }
