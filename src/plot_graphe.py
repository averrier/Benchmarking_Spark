#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import csv

_7workers_file_path = "/home/bastien/Documents/spark_project/results_7workers.csv"

with open(_7workers_file_path) as _7workers_file:
	dataToPlot = [[],[],[]]
	csv_reader = csv.reader(_7workers_file, delimiter=',')
	rows = []
	for row in csv_reader:
		rows.append(row)#1:

	for j in range(3):
		for k in range(3):
			dataToPlot[j].append([[],[]])
			for i in range(4):
				dataToPlot[j][-1][0].append(float(rows[i+7*j][1]))
				dataToPlot[j][-1][1].append(float(rows[i+7*j][3+k]))


	#dataToPlot[j].append([[],[]])
	

	for j in range(3):
		for k in range(3):
			dataToPlot[j].append([[],[]])
			dataToPlot[j][-1][0].append(float(rows[0+7*j][2]))
			dataToPlot[j][-1][1].append(float(rows[0+7*j][3+k]))
			for i in range(4, 7):
				dataToPlot[j][-1][0].append(float(rows[i+7*j][2]))
				dataToPlot[j][-1][1].append(float(rows[i+7*j][3+k]))

	xlabels = ["number of rows", "number of features"]
	ylabels = ["training time (s)", "prediction time (s)", "accuracy (%)"]
	ylabels_reg = ["training time (s)", "prediction time (s)", "r_mse (hours)"]
	
	print(dataToPlot[2])

	color = ["orange", "blue", "green"]

	for j in range(6):
		fig = plt.figure()
		for i in range(3):
			if(j%3 == 2 and i == 2):
				break
			x = dataToPlot[i][j][0]
			y = dataToPlot[i][j][1]
			if(j%3 == 2):
				y = [y_ * 100 for y_ in y]
			print(x,y)
			plt.scatter(x[::-1], y[::-1], color=color[i])
			plt.plot(x[::-1], y[::-1], color=color[i])
			plt.xlabel(xlabels[int(j/3)])
			plt.ylabel(ylabels[j%3])
		if(j%3 != 2):	
			plt.legend(['Logistic Regression', 'Random Forest', 'Linear Regression'], loc="best")
		else:
			plt.legend(['Logistic Regression', 'Random Forest'], loc="best")

		plt.savefig('/home/bastien/Documents/spark_project/figures/'+xlabels[int(j/3)]+'_'+ylabels[j%3]+'.png')
		plt.show()

		if(j%3 == 2):
			x = dataToPlot[2][j][0]
			y = dataToPlot[2][j][1]
			print(x,y)
			plt.scatter(x[::-1], y[::-1], color=color[i])
			plt.plot(x[::-1], y[::-1], color=color[i])
			plt.xlabel(xlabels[int(j/3)])
			plt.ylabel(ylabels_reg[j%3])
			plt.legend(['Linear Regression'], loc="best")
			plt.savefig('/home/bastien/Documents/spark_project/figures/'+xlabels[int(j/3)]+'_'+ylabels[j%3]+'_reg.png')
			plt.show()