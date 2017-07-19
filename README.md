# HeartDiseaseStudy

For this project I applied a logistic regression model to the Cleveland Heart Disease data set.

This was my "Project McNulty" in the Spring 2015 Metis Data Science Boot Camp. 

See blog post at [lucdemortier.github.io](http://lucdemortier.github.io/projects/3_mcnulty.html) for a description of the results.

iPython notebooks and other files used to generate the results and plots for the McNulty project:

1. **convert\_ssv\_to\_csv.py**: Converts a file with space-separated values into a file with comma-separated values.

1. **join\_files.py**: Joins files downloaded from the UC Irvine Machine Learning Repository into a single file for processing by the iPython notebook below.

1. **KNearestNeighbors.py**: Short program to select features by maximizing the accuracy of K-nearest neighbors classifier.

1. **HeartDiseaseProject.ipynb**: iPython notebook to read in the data, store them in a Pandas dataframe for initial processing and plots, and analyze with a logistic regression model.  Cells at the end of the notebook investigate naive Bayes, support vector machine, decision tree, and random forest classifiers to select features that maximize accuracy.  These methods were not pursued further however.

The initial processing steps of this study are as follows:

`curl -o data/cleveland14.csv https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data`

`curl -o data/hungarian14r.ssv https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/reprocessed.hungarian.data`

`curl -o data/switzerland14.csv https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data`

`curl -o data/long_beach_va14.csv https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data`

`python convert_ssv_to_csv.py hungarian14r`

`python join_files.py`

The output of join\_files.py is file data/heart\_disease\_all14.csv and is ready for processing by HeartDiseaseProject.ipynb.
