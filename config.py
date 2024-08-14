import os

# dataset name
dataset = 'ml-1m'
assert dataset in ['ml-1m', 'pinterest-20', 'yelp']

# paths
main_path = './Data/'

train_rating = os.path.join(main_path, '{}.train.rating'.format(dataset))
test_rating = os.path.join(main_path, '{}.test.rating'.format(dataset))
test_negative = os.path.join(main_path, '{}.test.negative'.format(dataset))
