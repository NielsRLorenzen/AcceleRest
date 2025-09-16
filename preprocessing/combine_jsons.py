import json
import os
import random

path = '/oak/stanford/groups/mignot/projects/actigraphy_fm/data/splits/'
dsets = ['stages/', 'tbi/', 'dreamt/']
trainset = []
valset = []
for dset in dsets:
    with open(path + dset + 'all_train_files.json', 'r') as f:
        trainset.extend(json.load(f))
    with open(path + dset + 'test_files.json', 'r') as f:
        valset.extend(json.load(f))
    print(len(trainset))
    print(len(valset))
with open(path + 'amazfit/' + 'test_files.json', 'r') as f:
    testset = json.load(f)

random.seed(0)
random.shuffle(trainset)
random.shuffle(valset)
random.shuffle(testset)

output_dir = path + 'apnea/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(f'{output_dir}/all_train_files.json', 'w') as f:
    json.dump(trainset, f)
with open(f'{output_dir}/val_files.json', 'w') as f:
    json.dump(valset, f)
with open(f'{output_dir}/test_files.json', 'w') as f:
    json.dump(testset, f)

with open(f'{output_dir}/all_train_files.json', 'r') as f:
    print(json.load(f)[:10])
with open(f'{output_dir}/val_files.json', 'r') as f:
    print(json.load(f)[:10])
with open(f'{output_dir}/test_files.json', 'r') as f:
    print(json.load(f)[:10])

# dsets = ['newcastle_left/', 'newcastle_right/', 'stages/', 'tbi/', 'dreamt/']
# trainset = []
# testset = []
# for dset in dsets:
#     with open(path + dset + 'all_train_files.json') as f:
#         trainset.extend(json.load(f))
#     with open(path + dset + 'test_files.json') as f:
#         testset.extend(json.load(f))
#     print(len(trainset))
#     print(len(testset))
    
# random.seed(0)
# random.shuffle(trainset)
# random.shuffle(testset)
# # output_dir = path + 'internal'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# with open(f'{output_dir}/all_train_files.json', 'w') as f:
#     json.dump(trainset, f)
# with open(f'{output_dir}/test_files.json', 'w') as f:
#     json.dump(testset, f)

# with open(f'{output_dir}/all_train_files.json', 'r') as f:
#     print(json.load(f)[:10])
# with open(f'{output_dir}/test_files.json', 'r') as f:
#     print(json.load(f)[:10])

# ext_testsets = ['amazfit/', 'sleepaccel/']
# ext_testset = []
# for dset in ext_testsets:
#     with open(path + dset + 'test_files.json') as f:
#         ext_testset.extend(json.load(f))
#     print(len(ext_testset))
# random.shuffle(ext_testset)

# trainset.extend(testset)
# random.shuffle(trainset)

# output_dir = path + 'external'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# with open(f'{output_dir}/all_train_files.json', 'w') as f:
#     json.dump(trainset, f)
# with open(f'{output_dir}/test_files.json', 'w') as f:
#     json.dump(ext_testset, f)
