import mdpsolver
import numpy as np
from itertools import product
import time

class NoHolesTetris:
    def __init__(self, width=4, height=4, num_pieces=1):
        self.width = width
        self.height = height
        self.num_pieces = num_pieces
        
        # Define the Tetris pieces
        self.pieces = self._define_pieces()
        
        # Build the MDP model
        self.mdl = mdpsolver.model()
        self._build_mdp()
        
    def _define_pieces(self):
        """Define the Tetris pieces and their possible placements."""
        pieces = []
        
        # I-piece
        pieces.append([
            # Horizontal I-piece (width 4, height 1)
            {
                'width': 4,
                'height': 1,
                'shape': [[1, 1, 1, 1]],
                'offsets': [(0, i) for i in range(self.width - 3)]
            },
            # Vertical I-piece (width 1, height 4)
            {
                'width': 1,
                'height': 4,
                'shape': [[1], [1], [1], [1]],
                'offsets': [(0, i) for i in range(self.width)]
            }
        ])
        
        if self.num_pieces > 1:
            # L-piece
            pieces.append([
                # L-piece orientation 1 (width 2, height 3)
                {
                    'width': 2,
                    'height': 3,
                    'shape': [[1, 0], [1, 0], [1, 1]],
                    'offsets': [(0, i) for i in range(self.width - 1)]
                },
                # L-piece orientation 2 (width 3, height 2)
                {
                    'width': 3,
                    'height': 2,
                    'shape': [[1, 1, 1], [1, 0, 0]],
                    'offsets': [(0, i) for i in range(self.width - 2)]
                },
                # L-piece orientation 3 (width 2, height 3)
                {
                    'width': 2,
                    'height': 3,
                    'shape': [[1, 1], [0, 1], [0, 1]],
                    'offsets': [(0, i) for i in range(self.width - 1)]
                },
                # L-piece orientation 4 (width 3, height 2)
                {
                    'width': 3,
                    'height': 2,
                    'shape': [[0, 0, 1], [1, 1, 1]],
                    'offsets': [(0, i) for i in range(self.width - 2)]
                }
            ])
        
        if self.num_pieces > 2:
            # O-piece (square)
            pieces.append([
                {
                    'width': 2,
                    'height': 2,
                    'shape': [[1, 1], [1, 1]],
                    'offsets': [(0, i) for i in range(self.width - 1)]
                }
            ])
        
        return pieces[:self.num_pieces]
        
    def _state_to_index(self, heights, piece_idx):
        """Convert a state (heights + next piece) to a unique index."""
        base_idx = 0
        multiplier = 1
        for h in heights:
            base_idx += h * multiplier
            multiplier *= (self.height + 1)
            
        return base_idx * self.num_pieces + piece_idx
        
    def _index_to_state(self, index):
        """Convert index back to state (heights + next piece)."""
        piece_idx = index % self.num_pieces
        board_idx = index // self.num_pieces
        
        heights = []
        for _ in range(self.width):
            heights.append(board_idx % (self.height + 1))
            board_idx //= (self.height + 1)
            
        return heights, piece_idx
    
    def _calculate_drop_height(self, heights, piece, offset_col):
        """Calculate drop height for a piece at given offset."""
        drop_height = 0
        for piece_row in range(len(piece['shape'])):
            for piece_col in range(len(piece['shape'][0])):
                if piece['shape'][piece_row][piece_col] == 1:
                    col = offset_col + piece_col
                    # The height at this column before placing the piece
                    current_height = heights[col]
                    # The height needed for this part of the piece
                    needed_height = drop_height + piece_row + 1
                    # Update drop_height if needed
                    if needed_height > current_height:
                        drop_height = needed_height - piece_row - 1
        
        return drop_height
    
    def _place_piece(self, heights, piece, offset_col, drop_height):
        """Place a piece on the board and return new heights."""
        new_heights = list(heights)
        
        # Update heights based on piece placement
        for piece_row in range(len(piece['shape'])):
            for piece_col in range(len(piece['shape'][0])):
                if piece['shape'][piece_row][piece_col] == 1:
                    col = offset_col + piece_col
                    height = drop_height + piece_row
                    if height >= new_heights[col]:
                        new_heights[col] = height + 1

        
        return new_heights
    
    def _count_lines_cleared(self, new_heights):
        """Count completed lines and update heights after clearing."""
        # Find heights where all columns are at the same level
        height_counts = {}
        for h in new_heights:
            if h in height_counts:
                height_counts[h] += 1
            else:
                height_counts[h] = 1
        
        lines_cleared = 0
        cleared_heights = []
        
        # Identify complete lines
        for height, count in height_counts.items():
            if count == self.width and height > 0:  # Complete line
                lines_cleared += 1
                cleared_heights.append(height)
        
        # No lines cleared, return original heights
        if lines_cleared == 0:
            return new_heights, 0
        
        # Sort cleared heights (important for proper shifting)
        cleared_heights.sort()
        
        # First, create a representation of the board as a 2D grid
        # Initialize with zeros
        board = [[0 for _ in range(self.width)] for _ in range(max(new_heights) + 1)]
        
        # Fill in the board based on heights
        for col in range(self.width):
            for row in range(new_heights[col]):
                board[row][col] = 1
        
        # Remove rows that are completely filled (lines to clear)
        for height in cleared_heights:
            # Remove the row at this height (0-indexed)
            board.pop(height - 1)
            # Add an empty row at the top
            board.append([0] * self.width)
        
        # Recalculate heights after clearing
        adjusted_heights = [0] * self.width
        for col in range(self.width):
            # Count cells from bottom up until we find an empty cell
            for row in range(len(board)):
                if board[row][col] == 1:
                    adjusted_heights[col] = row + 1
        
        return adjusted_heights, lines_cleared
    
    def _count_holes(self, new_heights):
        """Count holes between columns using height differences."""
        holes = 0
        for i in range(len(new_heights) - 1):
            # Calculate height difference between adjacent columns
            diff = new_heights[i] - new_heights[i+1]
            if diff > 1:  # Column i is higher, creating potential holes
                holes += diff - 1
            elif diff < -1:  # Column i+1 is higher, creating potential holes
                holes += -diff - 1
        
        # Also consider the "unevenness" of the board as a proxy for potential holes
        max_height = max(new_heights)
        unevenness = sum(max_height - h for h in new_heights) / 2
        
        return holes + unevenness
    
    def _get_actions(self, heights, piece_idx):
        """Get possible actions (placements) for the given state."""
        piece_orientations = self.pieces[piece_idx]
        valid_actions = []
        
        for orient_idx, piece in enumerate(piece_orientations):
            for offset_idx, (offset_row, offset_col) in enumerate(piece['offsets']):
                # Calculate drop height for this piece and offset
                drop_height = self._calculate_drop_height(heights, piece, offset_col)
                
                # Check if placement is valid (not too high)
                if drop_height + piece['height'] > self.height:
                    continue
                
                # Place the piece and get new board state
                new_heights = self._place_piece(heights, piece, offset_col, drop_height)

                # Clear lines and get updated heights
                new_heights, lines_cleared = self._count_lines_cleared(new_heights)
                
            
                holes = self._count_holes(new_heights)
                
                # Calculate total reward
                reward = lines_cleared
                
                # Check for game over
                if max(new_heights) > self.height:
                    continue
                
                action_info = {
                    'orientation': orient_idx,
                    'offset': (offset_row, offset_col),
                    'new_heights': tuple(new_heights),
                    'reward': reward,
                    'lines_cleared': lines_cleared,
                    'holes': holes
                }
                valid_actions.append(action_info)
        
        return valid_actions
    
    def _build_mdp(self):
        """Build the MDP representation of the game."""
        # Calculate total number of states
        total_states = self.num_pieces * (self.height + 1) ** self.width + 1  # +1 for terminal state
        terminal_state = total_states - 1
        
        print(f"Building MDP with approximately {total_states} states...")
        
        # Generate rewards and transitions in elementwise format
        rewards_elementwise = []
        transitions_elementwise = []
        
        state_count = 0
        action_count = 0
        
        # Enumerate all possible states
        for heights in product(range(self.height + 1), repeat=self.width):
            for piece_idx in range(self.num_pieces):
                state_idx = self._state_to_index(heights, piece_idx)
                
                # Get possible actions for this state
                actions = self._get_actions(heights, piece_idx)
                
                # If no valid actions, transition to terminal state with large negative reward
                if not actions:
                    rewards_elementwise.append([state_idx, 0, -self.height])  # Cost of losing
                    transitions_elementwise.append([state_idx, 0, terminal_state, 1.0])
                    continue
                
                # For each valid action
                for action_idx, action in enumerate(actions):
                    new_heights = action['new_heights']
                    reward = action['reward']
                    
                    # Add reward for this state-action pair
                    rewards_elementwise.append([state_idx, action_idx, reward])
                    
                    # Add transitions for each possible next piece
                    for next_piece_idx in range(self.num_pieces):
                        next_state_idx = self._state_to_index(new_heights, next_piece_idx)
                        transitions_elementwise.append([
                            state_idx, action_idx, next_state_idx, 1.0/self.num_pieces
                        ])
                    
                    action_count += 1
                
                state_count += 1
                if state_count % 1000 == 0:
                    print(f"Processed {state_count} states, {action_count} actions...")
        
        print(f"Total states processed: {state_count}")
        print(f"Total actions: {action_count}")
        print(f"Total transitions: {len(transitions_elementwise)}")
        
        # Configure the MDP model
        self.mdl.mdp(
            discount=0.99,
            rewardsElementwise=rewards_elementwise,
            tranMatElementwise=transitions_elementwise
        )
    
    def solve(self, algorithm="mpi", tolerance=1e-3, verbose=True):
        """Solve the MDP to find optimal policy."""
        start_time = time.time()
        
        self.mdl.solve(
            algorithm=algorithm,
            tolerance=tolerance,
            verbose=verbose
        )
        
        end_time = time.time()
        solution_time = end_time - start_time
        
        if verbose:
            print(f"Solution time: {solution_time:.4f} seconds")
        
        return solution_time
    
    
    def play_game(self, max_moves=50):
        """Simulate playing a game using the optimal policy."""
        # Initialize board with all zeros
        heights = tuple([0] * self.width)
        piece_idx = np.random.randint(0, self.num_pieces)
        
        total_reward = 0
        moves = 0
        total_lines = 0
        
        print(f"Starting game with piece {piece_idx}")
        print(f"Board: {heights}")
        
        while moves < max_moves:
            state_idx = self._state_to_index(heights, piece_idx)
            
            # Get all possible actions for this state
            actions = self._get_actions(heights, piece_idx)
            if not actions:
                print("Game over - no valid moves")
                break
            
            # Get action from policy
            action_idx = self.mdl.getAction(state_idx)
            if action_idx >= len(actions):
                print(f"Warning: Invalid action index {action_idx}, max is {len(actions)-1}")
                action_idx = 0
                
            # Apply the action
            action = actions[action_idx]
            heights = action['new_heights']
            reward = action['reward']
            total_reward += reward
            total_lines += action['lines_cleared']
            
            # Select next piece randomly
            piece_idx = np.random.randint(0, self.num_pieces)
            
            moves += 1
            
            # Print state
            orient_idx = action['orientation']
            offset = action['offset']
            print(f"Move {moves}, Piece: {piece_idx}, Orient: {orient_idx}, Offset: {offset}")
            print(f"Heights: {heights}, Lines: {action['lines_cleared']}, Holes: {action['holes']}, Reward: {reward}")
        
        print(f"Game finished after {moves} moves with total reward: {total_reward}")
        print(f"Total lines cleared: {total_lines}")
        return total_reward, total_lines

# Function to run experiments with various board sizes
def run_experiments():
    results = []
    
    # Start with small configurations
    configs = [
        (6, 3, 1),  # width, height, num_pieces
        (5, 4, 2),
        (7,3,1),
        # Uncommenting these will make the experiments take longer
        # (5, 4, 2),
        # (6, 4, 1),
    ]
    
    for width, height, num_pieces in configs:
        try:
            print(f"\nSolving {height}x{width} NoHoles Tetris with {num_pieces} pieces...")
            tetris = NoHolesTetris(width=width, height=height, num_pieces=num_pieces)
            solution_time = tetris.solve(verbose=True)
            
            # Play games to evaluate policy
            game_rewards = []
            game_lines = []
            for i in range(3):  # Play 3 games for evaluation
                print(f"\nPlaying game {i+1}:")
                reward, lines = tetris.play_game(max_moves=20)
                game_rewards.append(reward)
                game_lines.append(lines)
            
            avg_reward = sum(game_rewards) / len(game_rewards)
            avg_lines = sum(game_lines) / len(game_lines)
            
            results.append({
                'width': width,
                'height': height,
                'num_pieces': num_pieces,
                'solution_time': solution_time,
                'avg_reward': avg_reward,
                'avg_lines': avg_lines
            })
            
            print(f"Solved with average reward: {avg_reward}, average lines cleared: {avg_lines}")
        except Exception as e:
            import traceback
            print(f"Failed to solve {height}x{width} with {num_pieces} pieces:")
            print(traceback.format_exc())
            results.append({
                'width': width,
                'height': height,
                'num_pieces': num_pieces,
                'solution_time': None,
                'avg_reward': None,
                'avg_lines': None
            })
    
    return results

# Run the experiments
if __name__ == "__main__":
    results = run_experiments()
    
    # Print results in a table
    print("\nResults Summary:")
    print("Board Size | Pieces | Solution Time (s) | Avg Reward | Avg Lines")
    print("-" * 70)
    for r in results:
        if r['solution_time'] is not None:
            print(f"{r['height']}x{r['width']} | {r['num_pieces']} | {r['solution_time']:.4f} | {r['avg_reward']:.2f} | {r['avg_lines']:.2f}")
        else:
            print(f"{r['height']}x{r['width']} | {r['num_pieces']} | Failed | N/A | N/A")