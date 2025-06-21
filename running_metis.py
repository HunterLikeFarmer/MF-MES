import numpy as np
import time
from scipy.spatial import KDTree
from scripts.MultiFidelityBO import multifidelity_bayesian_opt as MFBO

class DatasetProcessor:
    def __init__(self, X_low, y_low, X_high, y_high, cost_low, cost_high):
        """
        Initialize with your dataset
        :param X_low: Low-fidelity input points (n_low x d)
        :param y_low: Low-fidelity outputs (n_low)
        :param X_high: High-fidelity input points (n_high x d)
        :param y_high: High-fidelity outputs (n_high)
        :param cost_low: Cost of low-fidelity evaluation
        :param cost_high: Cost of high-fidelity evaluation
        """
        self.X_low = X_low
        self.y_low = y_low
        self.X_high = X_high
        self.y_high = y_high
        self.cost_low = cost_low
        self.cost_high = cost_high
        
        # Create KD trees for fast lookup
        self.tree_low = KDTree(self.X_low)
        self.tree_high = KDTree(self.X_high)
        
        # Bounds computation
        all_X = np.vstack([self.X_low, self.X_high])
        self.bounds = np.array([np.min(all_X, axis=0), np.max(all_X, axis=0)])
        self.dim = self.X_low.shape[1]  # Input dimension
        
        # Fidelity features and costs
        self.fidelity_features = np.array([0, 1])
        self.costs = np.array([self.cost_low, self.cost_high])
        
        # Track available points
        self.available_low = np.ones(len(self.X_low), dtype=bool)
        self.available_high = np.ones(len(self.X_high), dtype=bool)
        self.all_y_high = []  # Track all high-fidelity observations
    
    def get_pool(self):
        """Get current available points for each fidelity"""
        return [
            self.X_low[self.available_low],
            self.X_high[self.available_high]
        ]
    
    def evaluate_point(self, x, fidelity):
        """
        Evaluate a point and return its value
        :param x: Input point (1 x d)
        :param fidelity: 0 for low, 1 for high
        :return: y value
        """
        if fidelity == 0:  # Low fidelity
            _, idx = self.tree_low.query(x, k=1)
            y = self.y_low[idx]
            self.available_low[idx] = False
        else:  # High fidelity
            _, idx = self.tree_high.query(x, k=1)
            y = self.y_high[idx]
            self.available_high[idx] = False
            self.all_y_high.append(y)
        return y
    
    def get_best_y(self):
        """Get best (minimum) high-fidelity value observed so far"""
        return min(self.all_y_high) if self.all_y_high else float('inf')

class MFMESOptimizer:
    def __init__(self, dataset, max_cost=1000, init_points_low=5, init_points_high=3):
        """
        Initialize MF-MES optimizer
        :param dataset: DatasetProcessor instance
        :param max_cost: Maximum evaluation budget
        :param init_points_low: Initial low-fidelity points
        :param init_points_high: Initial high-fidelity points
        """
        self.dataset = dataset
        self.max_cost = max_cost
        self.current_cost = 0
        self.iteration = 0
        self.best_y = float('inf')
        
        # Initialize with random points
        self.training_input = [[] for _ in range(2)]
        self.training_output = [[] for _ in range(2)]
        self._initialize_training(init_points_low, init_points_high)
        
        # Initialize MF-MES
        self.optimizer = self._create_optimizer()
    
    def _initialize_training(self, num_low, num_high):
        """Initialize with random design points"""
        pool = self.dataset.get_pool()
        
        # Select random low-fidelity points
        if num_low > 0 and len(pool[0]) > 0:
            indices = np.random.choice(len(pool[0]), min(num_low, len(pool[0])), replace=False)
            x_low = pool[0][indices]
            y_low = [self.dataset.evaluate_point(x, 0) for x in x_low]
            self.training_input[0] = x_low
            self.training_output[0] = y_low
            self.current_cost += len(x_low) * self.dataset.cost_low
        
        # Select random high-fidelity points
        if num_high > 0 and len(pool[1]) > 0:
            indices = np.random.choice(len(pool[1]), min(num_high, len(pool[1])), replace=False)
            x_high = pool[1][indices]
            y_high = [self.dataset.evaluate_point(x, 1) for x in x_high]
            self.training_input[1] = x_high
            self.training_output[1] = y_high
            self.current_cost += len(x_high) * self.dataset.cost_high
        
        # Update best y
        self.best_y = self.dataset.get_best_y()
        
        # Print initialization
        print("\n=== Initial Design ===")
        for i, x in enumerate(self.training_input[0]):
            print(f"LOW: x={x}, y={self.training_output[0][i]:.4f}, cost={self.dataset.cost_low}")
        for i, x in enumerate(self.training_input[1]):
            print(f"HIGH: x={x}, y={self.training_output[1][i]:.4f}, cost={self.dataset.cost_high}")
        print(f"Initial best y: {self.best_y:.4f}")
        print(f"Total initial cost: {self.current_cost:.2f}\n")
    
    def _create_optimizer(self):
        """Create MF-MES optimizer instance"""
        return MFBO.MultiFidelityMaxvalueEntropySearch(
            X_list=self.training_input,
            Y_list=self.training_output,
            eval_num=[len(x) for x in self.training_input],
            bounds=self.dataset.bounds,
            kernel_bounds=None,  # Auto-configure
            M=2,  # Low and high fidelity
            cost=self.dataset.costs,
            sampling_num=10,
            sampling_method="RFM",  # Random Feature Map
            model_name="MFGP",
            fidelity_features=self.dataset.fidelity_features,
            pool_X=self.dataset.get_pool(),
            optimize=True
        )
    
    def run_optimization(self):
        """Run the optimization loop"""
        print("Starting MF-MES optimization...")
        print(f"{'Iter':<5} {'Cost':<8} {'x':<30} {'Fidelity':<8} {'y':<10} {'Best y':<10}")
        print("-" * 70)
        
        while self.current_cost < self.max_cost:
            self.iteration += 1
            
            # Select next point to evaluate
            new_inputs, new_pool = self.optimizer.next_input_pool(self.dataset.get_pool())
            
            # Determine which fidelity was chosen
            if len(new_inputs[0]) > 0:
                fidelity = 0
                x = new_inputs[0][0]
                cost = self.dataset.cost_low
            else:
                fidelity = 1
                x = new_inputs[1][0]
                cost = self.dataset.cost_high
            
            # Evaluate the point
            y = self.dataset.evaluate_point(x, fidelity)
            self.current_cost += cost
            
            # Update best y if high-fidelity improvement
            if fidelity == 1 and y < self.best_y:
                self.best_y = y
            
            # Print iteration details
            fid_str = "LOW" if fidelity == 0 else "HIGH"
            print(f"{self.iteration:<5} {self.current_cost:<8.2f} {str(x):<30} {fid_str:<8} {y:<10.4f} {self.best_y:<10.4f}")
            
            # Update training data
            new_input_list = [[], []]
            new_output_list = [[], []]
            new_input_list[fidelity] = [x]
            new_output_list[fidelity] = [y]
            
            # Update optimizer
            self.optimizer.update(new_input_list, new_output_list)
            
            # Check termination
            if self.current_cost >= self.max_cost:
                print(f"\nOptimization completed! Reached max cost of {self.max_cost:.2f}")
                print(f"Final best y: {self.best_y:.4f}")
                break

if __name__ == "__main__":
    # Example dataset - REPLACE WITH YOUR ACTUAL DATA
    # Low-fidelity data
    X_low = np.random.rand(100, 2)  # 100 points, 2D
    y_low = np.sum(X_low**2, axis=1) + np.random.normal(0, 0.1, 100)
    
    # High-fidelity data
    X_high = np.random.rand(50, 2)
    y_high = np.sum(X_high**2, axis=1)
    
    # Costs
    cost_low = 1.0
    cost_high = 5.0
    
    # Create dataset processor
    dataset = DatasetProcessor(X_low, y_low, X_high, y_high, cost_low, cost_high)
    
    # Create and run optimizer
    optimizer = MFMESOptimizer(
        dataset=dataset,
        max_cost=50,  # Your evaluation budget
        init_points_low=3,  # Initial low-fidelity points
        init_points_high=2  # Initial high-fidelity points
    )
    
    optimizer.run_optimization()