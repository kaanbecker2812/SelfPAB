import warnings
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module="torchmetrics.utilities.imports",
)

import os
import argparse
import train
import src.config
import src.datasets
import torch
import numpy as np
import random
from ray import tune

def loso_cv(config, dataset_path=None, hopt=False):
    '''Starts a leave-one-out cross validation'''
    # Set all seeds:
    if type(config) == dict:
        config = src.config.Config(config)
        if hopt:
            config.WANDB = False
            config.FOLDS = 1
            config.VALID_SPLIT = 0.0
            config.NUM_GPUS = [0]
    print(type(config))
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # In LOSO test subject is used for valid
    valid_split = 'test' if config.FOLDS==0 else config.VALID_SPLIT
    print('valid_split=',valid_split)
    if config.WANDB:
        #ds_name = os.path.realpath(dataset_path).split('/')[-1]
        proj_name = 'EASE_'+config.PROJ_NAME
        run_name = config.ALGORITHM+'_'+config.DATASET
        src.utils.wandb_init(
            run_name=run_name,
            wandb_config=vars(config),
            entity='kaan-becker-technical-univeryity-munich',
            proj_name=proj_name,
            key=config.WANDB_KEY
        )
        all_test_true = []
        all_test_pred = []
    filenames = sorted(src.utils.get_filenames_of_dataset(dataset_path))
    if len(filenames) == 0:
        filenames = sorted(src.utils.get_filenames_of_dataset(dataset_path,
                                                              filetype='dat'))
    fold_performances = []
    for fold, train_files, test_files in src.utils.cv_split(filenames,
                                                            config.FOLDS,
                                                            config.SEED,
                                                            config.TEST_SPLIT):
        if type(valid_split)==float:
            config.VALID_SPLIT = random.sample(
                train_files,
                int(valid_split * len(train_files))
            )
        else:
            config.VALID_SPLIT = valid_split
        config.TEST_SUBJECTS = test_files
        print(f'No. of train subjects: {len(train_files)}')
        print(f'Test subject: {test_files[0].split("_")[2]}; No. of test subjects: {len(test_files)}')
        if config.WANDB:
            config.WANDB_GROUP = f'fold_{fold + 1}'
        best_model,test_cmat,best_logs,best_args = train.train(config,dataset_path,loso=True, fold_num=fold + 1)
        if config.WANDB:
            src.utils.log_cmat_metrics_to_wandb(
                log_cmat=test_cmat,
                log_name= f'per_Fold',
                class_names=config.class_names,
                metrics=['average_f1score',
                            'average_recall',
                            'average_precision',
                            'accuracy'
                        ])
            src.utils.log_cmat_metrics_to_wandb(
                log_cmat=test_cmat,#[test_filename],
                log_name= f'Fold_{fold+1}',
                class_names=config.class_names,
                metrics=['cmat'])
            src.utils.log_history_metrics_to_wandb(
                    metrics_dict=best_logs,
                    log_name=f'best_Fold_{test_files[0].split("_")[2]}'#test_filename,
                )
            all_test_true += list(test_cmat.y_true)#[test_filename].y_true)
            all_test_pred += list(test_cmat.y_pred)#[test_filename].y_pred)
        #fold_performances.append(np.mean([getattr(_c, config.EVAL_METRIC) for _c in test_cmat.values()]))
        fold_performances.append(getattr(test_cmat, config.EVAL_METRIC))
        if config.STORE_CMATS:
            to_store_path = f'{config.STORE_PATH}/reports/'
            if test_cmat is None:
                breakpoint()
            src.utils.save_intermediate_cmat(
                path=to_store_path,
                filename=f'best_Fold_{test_files[0].split("_")[2]}' + '_cmat.pkl',
                args=best_args,
                cmats={f'best_Fold_{test_files[0].split("_")[2]}': test_cmat},
                valid_subjects=[fold + 1]
            )
            best_model_path = os.path.join(f'{config.STORE_PATH}/models/',
                                           f'best_Fold_{test_files[0].split("_")[2]}_model.ckpt')
            #os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            # Save unwrapped model on CPU to avoid DDP/sharded tensor state_dict issues
            #_m = best_model.module if hasattr(best_model, "module") else best_model
            #torch.save(_m.cpu().state_dict(), best_model_path)
    final_CV_test_perf_mean = np.mean(fold_performances)
    final_CV_test_perf_std = np.std(fold_performances)
    print(f'Final {fold + 1}-fold CV {config.EVAL_METRIC}: {final_CV_test_perf_mean}({final_CV_test_perf_std})')
    if hopt:
        tune.report(score_mean=final_CV_test_perf_mean,
                    score_std=final_CV_test_perf_std,
                    score_name=config.EVAL_METRIC)

    if config.WANDB:
        src.utils.log_wandb_cmat(
            y_true=all_test_true,
            y_pred=all_test_pred,
            class_names=config.class_names,
            log_name='Total'
        )




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start LOSO CV.')
    parser.add_argument('-p', '--params_path', required=False, type=str,
                        help='params path with config.yml file',
                        default='/param/config.yml')
    parser.add_argument('-d', '--dataset_path', required=False, type=str,
                        help='path to dataset.', default=None)
    args = parser.parse_args()
    config_path = args.params_path
    # Read config
    config = src.config.Config(config_path)
    ds_path = args.dataset_path
    loso_cv(config, ds_path)
