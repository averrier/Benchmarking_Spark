#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import csv

file_path = "/home/bastien/Documents/spark_project/result_workers.csv"

with open(file_path) as file:
	dataToPlot = [[],[],[]]
	csv_reader = csv.reader(file, delimiter=',')
	rows = []
	for row in csv_reader:
		rows.append(row)

	rows.sort(key=lambda x: x[0])
	print(rows)

	x_workers = []
	y_trainTime = [[], [], []]
	y_predTime = [[], [], []]
	for i,row in enumerate(rows):
		if(i % 3 == 0):
			x_workers.append(float(row[0]))
		y_trainTime[i%3].append(float(row[4]))
		y_predTime[i%3].append(float(row[5]))

	#x_workers.sort()
	xlabels = "number of workers"
	ylabels = ["training time (s)", "prediction time (s)"]

	
	for j,metric in enumerate([y_trainTime, y_predTime]):
		fig = plt.figure()
		for i in range(3):
			x = x_workers
			y = metric[i]
			print(x,y)
			plt.scatter(x, y)
			plt.plot(x, y)
			plt.xlabel(xlabels)
			plt.ylabel(ylabels[j])
		plt.legend(['Logistic Regression', 'Random Forest', 'Linear Regression'], loc="best")
		plt.savefig('/home/bastien/Documents/spark_project/figures/'+xlabels+'_'+ylabels[j]+'.png')
		plt.show()

		# x = x_workers
		# y = metric[2]
		# print(x,y)
		# plt.scatter(x, y)
		# plt.plot(x, y)
		# plt.xlabel(xlabels)
		# plt.ylabel(ylabels[j])
		# plt.legend(['Linear Regression'], loc="lower right")
		# plt.show()