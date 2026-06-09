import os
import yaml
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt

class ExperimentManager:
    def __init__(self, data_yaml='ppe_data.yaml', model_name='yolov8n.pt'):
        self.data_yaml = data_yaml
        self.model_name = model_name
        self.results_dir = 'experiments'
        os.makedirs(self.results_dir, exist_ok=True)
        self.summary_file = os.path.join(self.results_dir, 'experiment_summary.csv')
        self.summary_data = []

    def run_experiment(self, name, **kwargs):
        print(f"\n>>> Running Experiment: {name} with args: {kwargs}")
        model = YOLO(self.model_name)
        
        # Default training parameters for this study
        train_args = {
            'data': self.data_yaml,
            'epochs': 20,  # Reduced for faster verification of the entire flow
            'imgsz': 640,
            'batch': 16,
            'project': self.results_dir,
            'name': name,
            'exist_ok': True,
            'device': 'cpu', # Explicitly use cpu for this env
        }
        train_args.update(kwargs)
        
        results = model.train(**train_args)
        
        # Extract metrics from the last epoch
        # results.results_dict contains the final metrics
        metrics = results.results_dict
        summary = {
            'experiment_name': name,
            'precision': metrics.get('metrics/precision(B)', 0),
            'recall': metrics.get('metrics/recall(B)', 0),
            'mAP50': metrics.get('metrics/mAP50(B)', 0),
            'mAP50-95': metrics.get('metrics/mAP50-95(B)', 0),
        }
        # Add the hyperparameter that was changed
        summary.update(kwargs)
        self.summary_data.append(summary)
        self.save_summary()
        return summary

    def save_summary(self):
        df = pd.DataFrame(self.summary_data)
        df.to_csv(self.summary_file, index=False)
        print(f"Summary updated: {self.summary_file}")

    def plot_results(self, param_name):
        df = pd.DataFrame(self.summary_data)
        # Filter rows that have the parameter
        param_df = df[df[param_name].notna()].sort_values(by=param_name)
        if param_df.empty:
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(param_df[param_name], param_df['mAP50'], marker='o', label='mAP50')
        plt.title(f'Performance vs {param_name}')
        plt.xlabel(param_name)
        plt.ylabel('mAP50')
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, f'trend_{param_name}.png'))
        plt.close()

if __name__ == "__main__":
    manager = ExperimentManager()
    
    # Step 1: Baseline
    print("--- Starting Step 1: Baseline ---")
    manager.run_experiment('baseline')
    
    # Step 2: Single-variable experiments (HSV)
    # Sampling points
    hsv_configs = {
        'hsv_h': [0.0, 0.03, 0.05], # default 0.015
        'hsv_s': [0.0, 0.3, 1.0],   # default 0.7
        'hsv_v': [0.0, 0.2, 0.8]    # default 0.4
    }
    
    for param, values in hsv_configs.items():
        print(f"\n--- Starting Step 2: {param} Experiments ---")
        for val in values:
            manager.run_experiment(f'step2_{param}_{val}', **{param: val})
        manager.plot_results(param)

    print("\nAll experiments in Step 1 & 2 completed!")
