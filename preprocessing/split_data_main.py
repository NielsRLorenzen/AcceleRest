import argparse
import glob
import json
import os
import random

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--data_files_path',
        type = str,
        help = 'Path to the directory containing the subject .h5 files.',
        default = '/oak/stanford/groups/mignot/mdige/results/takeda/preprocessed_h5/',
    )
    argparser.add_argument(
        '--output_dir',
        type=str,
        help = 'Directory to save the output h5 files',
        default = '/oak/stanford/groups/mignot/projects/actigraphy_fm/data/splits/takeda/',
    )
    argparser.add_argument(
        '--test_prcnt',
        type = float,
        help = 'Percent of files in --data_files_path to put into test set.',
        default = 1.0,
    )
    argparser.add_argument(
        '--n_cv',
        type = int,
        help = 'Number of cross-validation folds to create from the train set.',
        default = 0,
    )
    argparser.add_argument(
        '--val_prcnt',
        type = float,
        help = (
            'Percent of train files that will be set aside for validation.'
            'Only used if cv_split <= 1.'
        ),
        default = 0.0,
    )
    return argparser.parse_args()

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    files = glob.glob(args.data_files_path + '*.h5')
    random.seed(0)
    random.shuffle(files)
    test_files = files[:int(args.test_prcnt * len(files))]
    # Create train and test json files
    with open(f'{args.output_dir}/test_files.json', 'w') as f:
        json.dump(test_files, f)
    if args.test_prcnt == 1.0:
        return
    
    train_files = files[int(args.test_prcnt * len(files)):]
    if args.n_cv > 1:
        os.makedirs(args.output_dir + '/folds/')
        folds = [train_files[i::args.n_cv] for i in range(args.n_cv)]
        for i in range(args.n_cv):
            with open(f'{args.output_dir}/folds/fold_{i}_train_files.json', 'w') as f:
                json.dump([item for sublist in folds[:i] + folds[i+1:] for item in sublist], f)
            with open(f'{args.output_dir}/folds/fold_{i}_validation_files.json', 'w') as f:
                json.dump(folds[i], f)

    elif args.val_prcnt > 0.0:
        val_files = train_files[:int(args.val_prcnt * len(train_files))]
        with open(f'{args.output_dir}/val_files.json', 'w') as f:
            json.dump(val_files, f)
        train_files = train_files[int(args.val_prcnt * len(train_files)):]


    with open(f'{args.output_dir}/all_train_files.json', 'w') as f:
        json.dump(train_files, f)

if __name__ == '__main__':
    args = parse_args()
    main(args)