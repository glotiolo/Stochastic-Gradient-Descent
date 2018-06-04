import csv
import random 

def load_csv(filename):
    lines = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

def load_adult_data():
    return load_csv("adult-data.csv")


#Splits the file into two categories Training set and test set given flag
def load_data(flag):
	data = load_adult_data()
	random.shuffle(data)#shuffle data randomly
	mid_point = int(len(data)*0.7)#currently 70% training
	#30% test
	train_data = data[:mid_point]
	test_data = data[mid_point:]

	if (flag == "train"):
		return train_data

	elif(flag == "test"):
		return test_data




#TODO: Possibly use different data for training and validation
def load_adult_train_data():
    return load_data('train')

def load_adult_valid_data():
    return load_data('test')


   



