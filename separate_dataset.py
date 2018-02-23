import datahandler
import os

#orig_dataset = datahandler.Dataset('data/9600Hz/')
orig_dataset = datahandler.Dataset('data/333Hz/')
train_set, test_set = orig_dataset.disjunct_split(.01)

train_set.save('data/train.csv')
test_set.save('data/test.csv')
