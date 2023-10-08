from node import Node
import math
from parse import parse
import utility
import random
import matplotlib.pyplot as plt 

def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''
  TARGETCLASS = "Class"
  #Parse Examples file and store the dictionary 
  dataset = examples

  #Attributes list 
  attributes = list(dataset[0].keys())
  attributes.remove(TARGETCLASS)

  # Fix missing values 
  

  #Call the recursive ID3 function to train on data
  return recID3(dataset, attributes)

def recID3(examples, attributes):


  # Check for base cases
  if(len(examples) == 1 or len(utility.getUniqueValuesForAttribute(examples, "Class")) == 1):
    return Node(label = examples[0]["Class"])

  node = Node()

  #Find the best attribute 
  bestAttributeName = utility.findBestAttribute(examples, attributes)
  uniqueAttributeValues = utility.getUniqueValuesForAttribute(examples, bestAttributeName)
  node.attribute = bestAttributeName

  #Get the remaining Atrributes
  newAttributes = attributes[:]
  newAttributes.remove(bestAttributeName)

  #For each value of this best attrbiute :
  for value in uniqueAttributeValues:

    #Get subset if the dataset where all entries of attribute has the same value
    subDataset = utility.getDataWithAttValue(examples, bestAttributeName,value)

    #call ID3 recursive function with subset data and find best attribue in this subtree
    node.children[value] = recID3(subDataset, newAttributes)

  return node




def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''





def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''
  result = 0

  for example in examples:
    result += (evaluate(node, example) == example["Class"] * 1)

  return result / len(examples)

def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''
  while(node):
  
    if(node.label is not None):
      return node.label

    else:
      valueToLookFor = example[node.attribute]
      node = node.children[valueToLookFor]


def generateTrainingGraph(examples):
  # We need to take incremental Training samples from data set 
  testDataset = examples[:int(len(examples)/5)]
  examples = examples[int(len(examples)/5):]
  
  print(len(examples))
  numberTrainingSamples = range(10, len(examples), 5)
  
  Accuracies = []

  for numSamples in numberTrainingSamples:
    subset = [examples[i] for i in random.sample(range(len(examples)), numSamples)]
    
    trainedNode = ID3(subset, 0)
    Accuracies.append(test(trainedNode, [random.choice(testDataset) for _ in range(100)]))
  print(Accuracies)
  print(numberTrainingSamples)
  plt.plot(numberTrainingSamples, Accuracies)
  plt.plot()
  plt.xlabel('Number of Training Samples') 
  plt.ylabel('Accuracy') 
  plt.title('Training Samples Size Vs Accuracy') 
  plt.show() 

if __name__ == "__main__":
  examples = parse("house_votes_84.data")
  examples = utility.removeMissingValues(examples)
  generateTrainingGraph(examples)
