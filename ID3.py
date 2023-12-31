from node import Node
import math
from parse import parse
import utility
import random
import matplotlib.pyplot as plt 
from RandomForest import RandomForest
from collections import Counter

def impute_missing_value(dataset):
    attributes = list(dataset[0].keys())
    attributes.remove("Class")

    for attribute in attributes:
        # Get the values for the attribute.
        attribute_values = [example[attribute] for example in dataset]
        # Find the mode for non-missing values.
        non_missing_values = [value for value in attribute_values if value is not None and value != "?"]
        mode = max(set(non_missing_values), key=non_missing_values.count)

        # Update missing values in the dataset with the mode.
        for example in dataset:
            if example[attribute] is None or example[attribute] == "?":
                example[attribute] = mode

    return dataset


def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples. Each example is a dictionary of attribute:value pairs,
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
  dataset = impute_missing_value(dataset)

  #Call the recursive ID3 function to train on data
  return recID3(dataset, attributes)
    

def recID3(examples, attributes):

  # Check for base cases
  if len(examples) == 0:
     tree = Node()
     return tree
    
  if len(examples) == 1 or len(utility.getUniqueValuesForAttribute(examples, "Class")) == 1:
      return Node(label=examples[0]["Class"])
  
  else:
    node = Node()

    # Find the best attribute
    bestAttributeName = utility.findBestAttribute(examples, attributes)
    
    # Check if bestAttributeName is in attributes list before removal
    if bestAttributeName in attributes:
        uniqueAttributeValues = utility.getUniqueValuesForAttribute(examples, bestAttributeName)
        node.attribute = bestAttributeName

        # Get the remaining Attributes
        newAttributes = attributes[:]
        newAttributes.remove(bestAttributeName)

        # For each value of this best attribute:
        for value in uniqueAttributeValues:
            # Get subset of the dataset where all entries of attribute have the same value
            subDataset = utility.getDataWithAttValue(examples, bestAttributeName, value)

            # Call ID3 recursive function with subset data and find the best attribute in this subtree
            node.children[value] = recID3(subDataset, newAttributes)

    return node


# def prune(node, examples):
#     if node.label is not None:
#         return node

#     # Make a copy of the current tree
#     tree_copy = Node(label=node.label, attribute=node.attribute)
#     tree_copy.children = node.children.copy()

#     # Initialize the number of errors with the current tree
#     current_errors = len([example for example in examples if evaluate(node, example) != example["Class"]])

#     # Recursively prune the children
#     for value in node.children:
#         tree_copy.children[value] = prune(node.children[value], examples)

#     # Calculate the errors after pruning
#     pruned_errors = len([example for example in examples if evaluate(tree_copy, example) != example["Class"]])

#     # If pruning reduces errors, return the pruned tree; otherwise, return the original tree
#     if pruned_errors <= current_errors:
#         return tree_copy
#     else:
#         return node

def prune(node, examples, critical_value=0.5):

  """
  Takes in a trained tree and a validation set of examples. Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.

  Args:
    node: The tree node to prune.
    examples: The validation set of examples.
    critical_value: The critical value used to prune nodes.

  Returns:
    The pruned tree node.
  """

  # Calculate the accuracy of the current node on the validation data.
  accuracy = test(node, examples)

  # Prune the node if its accuracy is below the critical value and it is not a leaf node.
  if accuracy < critical_value and node.children:
    node.label = None
    for attVal,child in node.children.items():
      # Need to pass on the specific node wise dataset for accuracy prediction
      prune(child, utility.getDataWithAttValue(examples, node.attribute, attVal), critical_value)

  # Return the pruned node.
  return node


def test(node, examples):
  examples = impute_missing_value(examples)
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''
  result = 0

  for example in examples:
    result += (evaluate(node, example) == example["Class"] * 1)

  acc =  result / len(examples)
  return acc


def evaluate(node, example):
    while node:
        if node.label is not None:
            return node.label
        elif node.attribute is not None:
            valueToLookFor = example[node.attribute]
            node = node.children[valueToLookFor]
        else:
            return None  # Handle leaf nodes with no attribute


def randomForest(examples, trainSize, numTrees, default):
    
    randomForest = RandomForest()

    for i in range(numTrees):
       
      random.shuffle(examples)
      sliceDataset = examples[:trainSize]

      tree = ID3(sliceDataset, default)
      randomForest.trees.append(tree)

    #Now we have our random forest complete
    
    return randomForest
 

def generateTrainingGraph(examples):
  
  sampleSize = range(10, 300, 3)
  noPruneAcc = []
  pruneAcc = []

  for currentSampleSize in sampleSize:
    random.shuffle(examples)

    data = examples[:currentSampleSize]
    
    testDataset = examples[currentSampleSize:]

    validationDataset = testDataset[:(len(testDataset)*3)//4 ]

    forTest = testDataset[(len(testDataset)*3)//4:]
    hundredTests = [random.choice(forTest) for _ in range(101)]

    tree = ID3(data, 'democrat')
    acc = test(tree, hundredTests)
    noPruneAcc.append(acc)

    prunedTree = prune(tree, validationDataset, critical_value=0.8)
    acc = test(prunedTree, hundredTests)
    pruneAcc.append(acc)

  # Plot the graphs
  plt.figure(figsize=(10, 6))
  plt.plot(sampleSize, noPruneAcc, label="No Pruning")
  plt.plot(sampleSize, pruneAcc, label="Pruning")
  plt.xlabel('Number of Training Samples')
  plt.ylabel('Accuracy')
  plt.title('Training Samples Size Vs Accuracy')
  plt.legend()
  plt.show()

def EvaluateRandomForest(randomForest, testDataset):
  correctCount = 0
  for ex in testDataset:
    values = []
       
    for currentTree in randomForest.trees:
        values.append(evaluate(currentTree, ex))

    
    count = Counter(values)
    RFDecision = max(count, key=count.get)

    if(RFDecision == ex["Class"]):
        correctCount +=1
    
  return correctCount / len(testDataset) 

def compareRFDecisionTree(dataset, trainingSize, treeCount, default):
  random.shuffle(dataset)
  examples = dataset[:(9*len(dataset))//10]
  testDataset = dataset[(9*len(dataset))//10:]

  tree = ID3(examples, default)
  treeAcc = test(tree, testDataset)
  rdmForest = randomForest(examples, trainingSize, treeCount, default)
  forestAcc = EvaluateRandomForest(rdmForest, testDataset)

  return forestAcc, (forestAcc - treeAcc)/treeAcc * 100 

if __name__ == "__main__":
  examples = parse("house_votes_84.data")
  examples = impute_missing_value(examples)
  # examples = utility.removeMissingValues(examples)
  candy = parse("candy.data")

  samplesize = [10,20,30,40,50]
  treeCounts = [1,2,3,4,5,6,7,8,9,10]

  bestSample = 10
  bestCount = 1
  bestAcc = 0
  changeAccuracy = 0
  for s in samplesize:
     for t in treeCounts:
      print("Test for sample size : {} and number of Trees in Random Forest :{}".format(s, t))
      rfAccuracy, changeinAccuracy = compareRFDecisionTree(candy, s, t, 0)
      if(rfAccuracy > bestAcc):
         bestSample = s
         bestCount = t
         changeAccuracy = changeinAccuracy
         bestAcc = rfAccuracy
  
  print("Based on experiment, we have concluded that RF with {} sample size and {} trees performs the best with {} accuracy which was {} better than Tree".format(bestSample, bestCount, bestAcc, changeAccuracy))

  #generateTrainingGraph(examples)

