# Readme
# 
# Install:
# pip install sklearn
# pip install numpy
# pip install csv
# pip install copy
#
# Run:
# python this_filename
#

import csv
import numpy as np
import copy
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

# CONFIG START
FEATURES_CSV = 'features.csv'
TRAIN_FEATURES_CSV = 'train_features.csv'
TEST_FEATURES_CSV = 'test_features.csv'
SUBMIT_CSV = 'submission_rf.csv'
NO_OF_TREE_IN_RANDOM_FOREST = 800
SAMPLING_PERCENTAGE = 0.85
# CONFIG END


class RandomForest:
	def __init__(self, no_of_trees, sampling_percentage):
		self.sampling_percentage = sampling_percentage
		self.no_of_trees = no_of_trees
		self.trees = []
		for i in range(no_of_trees):
			tree = DecisionTreeClassifier(criterion='entropy', max_features='sqrt', random_state=i)
			self.trees.append(tree)

	def fit(self, samples, outcomes):
		sample_size_of_each_tree = int(len(samples) * self.sampling_percentage) 
		for k in range(self.no_of_trees):
			print ' > Fitting', k+1, 'of', self.no_of_trees
			x, y = shuffle(samples, outcomes, random_state=k*10)
			self.trees[k].fit(x[:sample_size_of_each_tree], y[:sample_size_of_each_tree])

	def predict_proba(self, samples):
		sample_size = len(samples)
		total = None
		for k in range(self.no_of_trees):
			print ' > Predicting', k+1, 'of', self.no_of_trees
			d = self.trees[k].predict_proba(samples)
			if (total == None):
				total = d
			else:
				for i in range(sample_size):
					for j in range(len(d[i])):
						total[i][j] += d[i][j]
		for i in range(sample_size):
			for j in range(len(total[i])):
				total[i][j] /= self.no_of_trees
		return total

class Data:
	def __init__(self, features, data, outcomes = []):
		self.features = copy.deepcopy(features)
		self.data = copy.deepcopy(data)
		self.outcomes = copy.deepcopy(outcomes)

	def drop(self, feature):
		index = self.features.index(feature)
		self.features.remove(feature)
		return [x.pop(index) for x in self.data]

	def boolean_to_value(self, feature):
		index = self.features.index(feature)
		for x in self.data:
			if (x[index] == 'True'):
				x[index] = 1
			else:
				x[index] = 0

	def fill_empty(self, feature):
		index = self.features.index(feature)
		for i in range(len(self.data)):
			if (self.data[i][index] == ''):
				self.data[i][index] = -1e30

	def fill_all_empty(self):
		for x in self.features:
			self.fill_empty(x)

	def string_to_value(self, feature):
		index = self.features.index(feature)
		values = [x[index] for x in self.data] 
		unique_values = list(set(values))
		for x in self.data:
			x[index] = unique_values.index(x[index])

# load the sample data
def load_data(filename):
	with open(filename, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		features = reader.next()
		data = []
		for row in reader:
			data.append(row)
		sample = Data(features, data)
		bidder_ids = sample.drop('bidder_id')
		for i in range(len(sample.data)):
			sample.data[i] = [float(y) for y in sample.data[i]]
		sample.outcomes = [int(x) for x in sample.drop('outcome')]
	return sample, bidder_ids

def write_data(filename, fieldnames, data):
	with open(filename, 'w') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		for row in data:
			obj = {}
			for i in range(len(fieldnames)):
				obj[fieldnames[i]] = row[i]
			writer.writerow(obj)

def run():
	# load data
	print 'Read the train file: ', TRAIN_FEATURES_CSV
	train, train_bidder_ids = load_data(TRAIN_FEATURES_CSV)
	print 'Total no. of record in ', TRAIN_FEATURES_CSV, ': ', len(train.data)

	# load test data
	print 'Read the test file: ', TEST_FEATURES_CSV
	test, test_bidder_ids = load_data(TEST_FEATURES_CSV)
	print 'Total no. of record in ', TEST_FEATURES_CSV, ': ', len(test.data)
	print 'Total no. of bidder ids in ', TEST_FEATURES_CSV, ': ', len(test_bidder_ids)

	# build a random forest
	print 'Bulid Random Forest'
	rf = RandomForest(NO_OF_TREE_IN_RANDOM_FOREST, SAMPLING_PERCENTAGE)
	rf.fit(train.data, train.outcomes)

	# run prediction on each test case
	print 'Prediction on test case'
	test_predict_outcomes = rf.predict_proba(test.data)
	test_predict_outcomes = [v[1] for v in test_predict_outcomes]

	print 'Prediction result'
	results = []
	for i in range(len(test_bidder_ids)):
		results.append([test_bidder_ids[i], test_predict_outcomes[i]])

	# output 
	print 'Output result to file: ', SUBMIT_CSV
	write_data(SUBMIT_CSV, ['bidder_id', 'prediction'], results)

	print 'END'

def pre_process():
	print 'Pre process the', FEATURES_CSV
	with open(FEATURES_CSV, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		features = reader.next()
		features[0] = 'bid_id'
		data = []
		for row in reader:
			data.append(row)

		sample = Data(features, data)
		sample.drop('bid_id')
		sample.string_to_value('most_common_country')
		sample.boolean_to_value('payment_account_prefix_same_as_address_prefix')
		sample.boolean_to_value('sleep')
		sample.fill_all_empty()

	outcome_index = sample.features.index('outcome')
	print 'Save to', TRAIN_FEATURES_CSV
	write_data(TRAIN_FEATURES_CSV, sample.features, [x for x in sample.data if int(float(x[outcome_index])) >= 0])
	print 'Save to', TEST_FEATURES_CSV
	write_data(TEST_FEATURES_CSV, sample.features, [x for x in sample.data if int(float(x[outcome_index])) == -1])

#Program Start
pre_process()
run()
#Program End
