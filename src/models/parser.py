def base_parser():
        
    import argparse
    parser = argparse.ArgumentParser()

    # General things
    parser.add_argument('--mode',  type=str,
                        default='train',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--test', type=bool,
                        default=False)
    parser.add_argument('--torch_tensor', type=bool,
                        default=True)
    parser.add_argument('--dataset', type=str,
                        default='GER',
                        choices=['GER'])
    parser.add_argument('--GPU', type=str, default='-1',
                        help='Set -1 for CPU running')
    parser.add_argument('--seed', type=int,
                        default=2)
    parser.add_argument('--train_dir', type=str,
                        default='/home/jbotz/Projects/aiolos_lstm/data')
    parser.add_argument('--data_name', type=str,
                        default='cases_daily_obs.csv')
    parser.add_argument('--save_path', type=str,
                        default='/home/jbotz/Projects/aiolos_lstm/models')
    parser.add_argument('--save_path_predictions', type=str,
                        default='/home/jbotz/Projects/aiolos_lstm/models/Predictions')
    parser.add_argument('--save_path_scores', type=str,
                        default='/home/jbotz/Projects/aiolos_lstm/models/Scores')
    parser.add_argument('--exp_name', type=str,
                        default='debug')
    parser.add_argument('--size_SW', type=int,
                        default=7,
                        help='Fitting Window Size')
    parser.add_argument('--step_SW', type=int,
                        default=1,
                        help='Step of Sliding Window')
    parser.add_argument('--size_PW', type=int,
                        default=7,
                        help='Prediction Window Size')
    parser.add_argument('--tuning', type=str,
                        default=None)
    parser.add_argument('--num_trials', type=int,
                        default=10)
    parser.add_argument('--scaled', type=bool,
                        default=False,
                        help='Standard Scaling')
    
    # rf parameters
    parser.add_argument('--rf_estimators', type=int,
                        default=200)
    parser.add_argument('--rf_max_depth', type=int,
                        default=None)
    parser.add_argument('--rf_min_samples_split', type=int,
                        default=2)
    parser.add_argument('--rf_min_samples_leaf', type=int,
                        default=1)                    

    # xg parameters
    parser.add_argument('--xg_learning_rate', type=float,
                        default=0.3)
    parser.add_argument('--xg_max_depth', type=int,
                        default=6)
    parser.add_argument('--xg_reg_lambda', type=int,
                        default=1)
    parser.add_argument('--xg_reg_alpha', type=int,
                        default=0) 

    parser.add_argument('--seasonality', type=bool,
                        default=False)
    # Metadata
    parser.add_argument('--hs_meta', type=int,
                        default=32) 
    parser.add_argument('--l_lstm_meta', type=int,
                        default=2) 

    parser.add_argument('--search_space')
    # Specific training parameters
    parser.add_argument('--period', type=int,
                        default=70)
    parser.add_argument('--iterations', type=int,
                        default=140)
    
    parser.add_argument('--iteration_start', type=int,
                        default=0)
    parser.add_argument('--iteration_end', type=int,
                        default=140)
    
    parser.add_argument('--batch_size', type=int,
                        default=16)
    parser.add_argument('--inp_dim', type=int,
                        default=1)
    parser.add_argument('--hs_dim', type=int,
                        default=64)
    parser.add_argument('--hs_ens_dim', type=int,
                        default=32)
    parser.add_argument('--lr_ens', type=float,
                        default=0.01)

    parser.add_argument('--type_pred', type=str,
                        default='selection',
                        choices=['selection', 'stacking'])
    parser.add_argument('--num_layers', type=int,
                        default=1,
                        choices=[1,2,3])
    parser.add_argument('--meta_lookback', type=int,
                        default=2,
                        choices=[1,2,3,4])
    parser.add_argument('--num_hidden_layers', type=int,
                        default=0,
                        choices=[0,1,2])
    
    parser.add_argument('--lr', type=float,
                        default=0.005)
    parser.add_argument('--num_epochs', type=int,
                        default=100)
    parser.add_argument('--epoch_init', type=int,
                        default=1)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--save_fig_freq', type=int, default=10)
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers')
    parser.add_argument('--patience', type=int,
                        default=25)
    parser.add_argument('--from_best', type=bool,
                        default=False)

    config = parser.parse_args()
    return config