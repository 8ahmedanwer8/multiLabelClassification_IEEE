import tensorflow as tf
import pickle
import numpy as np 
import statistics
import pandas as pd 
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score

pickle_in = open('X_test', 'rb')
X_test = pickle.load(pickle_in)

pickle_in = open('Y_test', 'rb')
Y_test = pickle.load(pickle_in)

model = tf.keras.models.load_model("Multilabel Model Version-2")

def evaluate_model(X_test, Y_test):
	pred = model.predict(X_test)
	actual = Y_test
	actual_list = actual.values.tolist()
	print(len(actual.iloc[0]))

	pred_label1 = pred[0]
	pred_label2 = pred[1]

	# print(pred_label1)
	# print(pred_label2)
	pred_temp_1 = []
	pred_temp_2 = []
	for item in pred_label1:
		temp = np.argmax(item)
		temp = temp + 1
		pred_temp_1.append(temp)

	for item in pred_label2:
		if item> 0.5:
			item = 1
		elif item< 0.5:
			item = 0
		else:
			item = "error"
		pred_temp_2.append(item)

	pred_list = zip(pred_temp_1, pred_temp_2)
	pred_list = list(pred_list)

	c=0
	hamming_loss_list = []
	exact_match_ratio_list = []

	while c < len(actual.iloc[:,0]):
		hamming_losss = hamming_loss(actual_list[c], pred_list[c])
		exact_match_ratio = accuracy_score(actual_list[c], pred_list[c])
		# print("-----------------------------")
		# print("hamming_loss: ", hamming_losss)
		# print("exact_match_ratio: ", exact_match_ratio)
		# print("Predicted: ", pred_list[c])
		# print("Actual: ", actual_list[c])
		# print("-----------------------------")

		hamming_loss_list.append(hamming_losss)
		exact_match_ratio_list.append(exact_match_ratio)
		c = c+1
	
	avg_hamming_loss = statistics.mean(hamming_loss_list)
	avg_exact_match_ratio = statistics.mean(exact_match_ratio_list)

	# print(hamming_loss_list)
	# print(exact_match_ratio_list)
	sd_hamming_loss = statistics.stdev(hamming_loss_list)
	sd_exact_match_ratio = statistics.stdev(exact_match_ratio_list)
	print("sd for hamming_loss", sd_hamming_loss)
	print("sd for exact_match_ratio", sd_exact_match_ratio)
	print("avg for hamming_loss", avg_hamming_loss)
	print("avg for exact_match_ratio", avg_exact_match_ratio)

	return sd_hamming_loss, sd_exact_match_ratio, avg_hamming_loss, avg_exact_match_ratio

evaluate_model(X_test, Y_test)
