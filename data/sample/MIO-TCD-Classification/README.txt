The 2017 MIO-TCD classification dataset contains 648,959 images devided into 11
categories, namely :

	articulated_truck
	bicycle
	bus
	car
	motorcycle
	non-motorized_vehicle
	pedestrian
	pickup_truck
	single_unit_truck
	work_van
	background

The dataset contains 519,164 training images (80%) and 129,795 testing images (20%)

The goal of this challenge is to **correctly label each image**.  The output
of your method shall to be put in a csv format such as 'gt_train.csv',
'your_results_test.csv', or 'your_results_train.csv' provided with the dataset.
'gt_train.csv' contains ground truth while 'your_results_test.csv' and
'your_results_train.csv' contains a random class assignment to the training and
the testing images.

NOTE: Python code is available online to help you play around with our dataset:
    tcd.miovision.com/challenge/dataset/

You may run the following command to parse the dataset and produce a valide
csv file :

> python parse_classification_dataset.py ./train/ your_results_train.csv
or
> python parse_classification_dataset.py ./test/ your_results_test.csv

You may also measure your training score with the 'classification_evaluation.py'
python code. For this, you only need to run the following command in a terminal

> python classification_evaluation gt_train.csv your_results_train.csv


That code was developed and tested with python 3.5.2 on Linux.

