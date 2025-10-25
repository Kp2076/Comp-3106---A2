
import pandas as pd
import numpy as np
import math

def naive_bayes_classifier(dataset_filepath, snake_measurements):
  # load dataset
  dataset = pd.read_csv(dataset_filepath, header=None)
  dataset.columns = ['class', 'length', 'weight', 'speed']
  
  classes = ['anaconda', 'cobra', 'python']
  features = ['length', 'weight', 'speed']
  
  total_data_points = len(dataset)

  #seperate data by class
  class_data = {cls: dataset[dataset['class'] == cls].drop(columns='class') for cls in classes}

  # store mean and std values for each class and feat 
  mean = {}
  std = {}
  prior_probability = {}

  # Compute mean, std for each (cls,feature) pair and probability of each class
  for cls in classes:
    for feature in features:
       temp = class_data[cls][feature]
       n = len(temp)
       tempMean = np.sum(temp)/n
       tempStd = np.sqrt(np.sum((temp - tempMean)**2) / (n-1))
       mean[(cls, feature)] = tempMean
       std[(cls,feature)] = tempStd
    prior_probability[cls] = len(class_data[cls]) / total_data_points 
    

  # Gaussian probability 
  def gaussian(cls, feature, x):
    curStd = std[(cls, feature)]
    curVariance = curStd**2
    curMean = mean[(cls, feature)]

    exponent = np.exp((-1/2)*(((x-curMean)/(curStd))**2))
    constant = 1 / np.sqrt(2 * np.pi * curVariance)
    return constant * exponent

  # Posterior probabilities
  posterior_probality = {}
  sum_prob_evidence = {}
  for cls in classes:
    sumProduct = 1
    for i, feature in enumerate(features):
      tempProb = gaussian(cls, feature, snake_measurements[i])
      posterior_probality[(feature,cls)] = tempProb
      sumProduct *= tempProb
    sum_prob_evidence[cls] = sumProduct * prior_probability[cls]
  
  # p(e) ----> P(lenght = x1 ^ weight = x2 ^ speed = x3) = sum[sumProduct(P(ek|hj))*P(hj)]
  prob_evidence = sum(sum_prob_evidence.values())
  
  print(prob_evidence)
  class_probabilities = []
  for cls in classes:
    class_probabilities.append(sum_prob_evidence[cls] / prob_evidence)

  # most likely class
  most_likely_class = np.argmax(class_probabilities)

  class_probabilities = np.array(class_probabilities).tolist()

  # dataset_filepath is the full file path to a CSV file containing the dataset
  # snake_measurements is a list of [length, weight, speed] measurements for a snake

  # most_likely_class is a string indicating the most likely class, either "anaconda", "cobra", or "python"
  # class_probabilities is a three element list indicating the probability of each class in the order [anaconda probability, cobra probability, python probability]

  return classes[most_likely_class], class_probabilities

##### TESTING #######

print("Test 0")
print(naive_bayes_classifier("./Examples/Example0/dataset.csv", [350, 42, 13]))

print("Test 1")
print(naive_bayes_classifier("./Examples/Example1/dataset.csv", [390, 28, 13]))

print("Test 2")
print(naive_bayes_classifier("./Examples/Example2/dataset.csv", [340, 26, 12]))

print("Test 3")
print(naive_bayes_classifier("./Examples/Example3/dataset.csv", [350, 42, 13]))

print("Test For 5")
print(naive_bayes_classifier("./Examples/Example0/dataset.csv", [350, 40, 15]))
print(naive_bayes_classifier("./Examples/Example0/dataset.csv", [310, 20, 12]))
print(naive_bayes_classifier("./Examples/Example0/dataset.csv", [392, 55, 19]))
print(naive_bayes_classifier("./Examples/Example0/dataset.csv", [315, 32, 18]))

