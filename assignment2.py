
import pandas as pd
import numpy as np
import math

def naive_bayes_classifier(dataset_filepath, snake_measurements):
  # load dataset
  x = pd.read_csv(dataset_filepath, header=None)
  x.columns = ['class', 'length', 'weight', 'speed']
  
  classes = ['anaconda', 'cobra', 'python']
  features = ['length', 'weight', 'speed']
  
  #seperate data by class
  class_data = {cls: x[x['class'] == cls] for cls in classes}
    
  # store mean and std values for each class and feat 
  mean = {}
  std = {}

  # Compute mean and std values
  for cls in classes:
    for feature in features:
       std.add((class,feature),sum((x_i - mean(cls, feature))**2) / (len(x) - 1))
  
  #Prior probability
  total = len(x)
  prior_probability = {cls: len(class_data[cls]) / total for cls in classes}

  # Gaussian probability 
  def gaussian(feature_x, mean_y, sigma):
    variance = sigma**2
    exponent = np.exp(-((feature_x - mean_y)**2) / (2 * variance))
    constant = 1 / np.sqrt(2 * np.pi * variance)
    return constant * exponent

  # Posterior probabilities
  posteriors = {}
    #calculation 

  # Normalize probabilities
  total_prob = sum(posteriors.values()) if posteriors else 1  
  normalized_probs = {cls: posteriors.get(cls, 0) / total_prob for cls in classes}

  # most likely class
  most_likely_class = max(normalized_probs, key=normalized_probs.get, default=None)

  # Return results in required order
  class_probabilities = [normalized_probs[cls] for cls in classes]



  #calculate mean and std

  # dataset_filepath is the full file path to a CSV file containing the dataset
  # snake_measurements is a list of [length, weight, speed] measurements for a snake

  # most_likely_class is a string indicating the most likely class, either "anaconda", "cobra", or "python"
  # class_probabilities is a three element list indicating the probability of each class in the order [anaconda probability, cobra probability, python probability]

  return most_likely_class, class_probabilities

