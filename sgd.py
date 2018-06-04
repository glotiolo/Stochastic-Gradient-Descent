# based on an assignment by Joe Redmon
#The program uses Stochastic gradient descent to predict the salaries of individuals 
#i.e whether they earn more than 50k or less.

from math import exp
import random
from operator import mul,add,sub
from data import load_adult_train_data, load_adult_valid_data

#Performs logistic function 
def logistic(x):       
    return  1/(1+ exp(-x))


#multiply two vectors 
def dot(x, y):
    prod_list = list(map(mul,x,y))
    return sum(prod_list)

#Function makes a prediction given a training example and a model
def predict(model, point):
    features = point["features"]
    h_theta = dot(features,model)
    return logistic(h_theta)


#Function calculates the accuracy of predictions
def accuracy(data, predictions):
    correct = 0
    for i in range(len(predictions)):
        comp = 0
        #Can only be one if its greater than 0.5
        if predictions[i] >= 0.5:
            comp = 1
        point = data[i]
        label = point['label']
        if(label== comp):
            correct = correct + 1
    return float(correct)/len(data)


#Function updates the model for every point checked
def update(model, point, alpha, lambd):
    h_theta = predict(model,point)
    error_term = h_theta - point['label']
    m = len(point['features'])
    expr_1 = [0]*m
    expr_1[0]= alpha * point['features'][0] * error_term
    model[0] = model[0] - expr_1[0]
    for i in range(1,m):
        expr_1[i] = alpha* ((point['features'][i] * error_term) + 
                    (model[i] * lambd))
        model[i] = model[i]-expr_1[i]


#initialize random set of theta    
def initialize_model(k):
    return [random.gauss(0, 1) for x in range(k)]



#Train model 
def train(data, epochs, alpha, lambd):
    m = len(data) # number of training examples
    n = len(data[0]['features']) # number of features (+ 1)
    model = initialize_model(n)
    minimizing_model = model
    best_accuracy = 0
    for i in range(epochs):
        for j in range(m):
            rand_index = random.randint(0,m-1)
            chosen_point = data[rand_index]
            #update the model 
            update(model,chosen_point,alpha,lambd)

        predictions = [predict(model, p) for p in data]
        accu = accuracy(data, predictions)

        #select the best accuracy value we have seen so far
        if (accu > best_accuracy):
            best_accuracy = accu
            minimizing_model = model
        print("")
        print("Validation Accuracy for epoch " + str(i+1) + ":", accu)    
    return minimizing_model

 
#Extracts features from the dataset      
def extract_features(raw):
    #Function extracts features from a dataset

    developed_countries = [ 'United-States','England','Canada',
        'Germany','Japan','Italy','Portugal', 'Ireland', 'France',
        'Scotland','Hong','Holand-Netherlands',]
    high_income_occupations = ['Sales', 'Exec-managerial','Prof-specialty','Tech-support','Protective-serv']

    govt_employment = ['Local-gov', 'State-gov','Federal-gov']

    data = []
    for r in raw:
        features = []
        features.append(1.0)
        age_val = float(r['age'])

        #This two groups have least proability to earn >50k
        features.append(age_val > 70 or age_val < 20)

        #Added a cubic function to education since
        #education doesn't have a perfect linear reationship to income
        features.append(float(r['education_num'])/20)
        features.append((float(r['education_num'])/20)**3)

        features.append(float(r['marital'] == 'Married-civ-spouse'))
        features.append(float(r['sex'] == 'Male'))

        #Hrs-worked has a positive correlation to incmome but levels off 
        #after some point
        features.append(float(r['hr_per_week'])/100) 
        features.append((float(r['hr_per_week'])/100)**2)

        #classified the occupation into professions with higher probability
        #and occupations with lower probability
        features.append(float(r['occupation'] in high_income_occupations))
        features.append(float(r['type_employer'] in govt_employment))


        point = {}
        point['features'] = features
        point['label'] = int(r['income'] == '>50K')
        data.append(point)
    return data





def submission(data):
    return train(data, epochs=40, alpha=0.0005, lambd=0)