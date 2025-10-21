
import pandas as pd
import numpy as np
import math

def naive_bayes_classifier(dataset_filepath, snake_measurements):
  x = pd.read_csv(dataset_filepath, header=None)
  x.columns = ['class', 'length', 'weight', 'speed']
  

  mean = set()
  #seperate data by class
  classes = ['anaconda', 'cobra', 'python']
  class_data = {}
  for cls in classes:
    class_data[cls] = x[x['class'] == cls]
    mean.add((cls, feature), meanValue)
    
  #standard deviation
  std = set()
  for cls in classes:
    for feature in features:
       std.add((class,feature),sum((x_i - mean(cls, feature))**2) / (len(x) - 1))
  
  
  #Prior probability
  total = len(x)
  prior_probability = {}
  for cls in classes:
    prior_probability[cls] = len(class_data[cls]) / total

  def gaussian(feature_x, mean_y, sigma):
    variance = sigma**2
    exponent = np.exp(-((x - mu)**2) / (2 * variance))
    constant = 1 / (np.sqrt(2 * np.pi * variance))
    return constant * variance



  
  
  #calculate mean and std

  # dataset_filepath is the full file path to a CSV file containing the dataset
  # snake_measurements is a list of [length, weight, speed] measurements for a snake

  # most_likely_class is a string indicating the most likely class, either "anaconda", "cobra", or "python"
  # class_probabilities is a three element list indicating the probability of each class in the order [anaconda probability, cobra probability, python probability]

  return most_likely_class, class_probabilities

