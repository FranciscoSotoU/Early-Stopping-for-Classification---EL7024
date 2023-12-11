import argparse
import os
from binary_experiment.base_experiment import BaseExperiment
from hd_experiment.HD_experiment import BaseExperimentHD
directory = os.getcwd()
if __name__ =='__main__':
    
    parser = argparse.ArgumentParser(description='Experiment')
    
    #Description and paths
    parser.add_argument('--exp_type', type = int, default = 1, help = 'Experiment type')
    parser.add_argument('--exp_path', type=str, default=os.path.join(directory,'experiments/'), help='where experiment are saved')
    parser.add_argument('--exp_name', type=str, default='experiment', help='Experiment Name')

    #Data
    parser.add_argument('--data_path', type=str, default=os.path.join(directory,'data/'), help='where data is saved')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--p', type=float, default=0.25, help='Probability of noise')
    parser.add_argument('--seed', type=int, default=42, help='Seed for data generation')

    #Model
    parser.add_argument('--hidden_dim', type=int, default=3, help='Hidden dimension')

    #Training
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')

    args = parser.parse_args()

    if args.exp_type == 0:
        base = BaseExperiment
    else:
        base = BaseExperimentHD
    experiment = base(args)
    
    experiment.run()


