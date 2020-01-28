# Benchmarking - Spark

We used Spark to study scalabilty of machine learning algorithms.

An open source dataset,the adult dataset, has been used to predict income, the number of working hours per week and the native country of a person. To do that, we trained different kind of machine learning models using the pyspark library such as linear regression, logistic regression and random forest.

In the git repo, you will find a video of presentation and the scripts used.
- the "tests_script.py" file is used to run all the tests over the dataset
- the "plot-graph.py" file is used to plot the graphs for all the tests with a fixed number of workers
- the "plot_graph_workers.py" file is used to plot the graphs for the execution time tests with a number of workers increasing
