from node import Node
import math
from parse import parse
import utility
import random
import matplotlib.pyplot as plt 



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

  return result / len(examples)


def evaluate(node, example):
    while node:
        if node.label is not None:
            return node.label
        elif node.attribute is not None:
            valueToLookFor = example[node.attribute]
            node = node.children[valueToLookFor]
        else:
            return None  # Handle leaf nodes with no attribute



def generateTrainingGraph(examples):
  # We need to take incremental Training samples from data set 
  testDataset = examples[:int(len(examples)/5)]
  examples = examples[int(len(examples)/5):]
  
  numberTrainingSamples = range(10, len(examples), 5)

  accuraciesNoPruning = []
  accuraciesPruning = []

  for numSamples in numberTrainingSamples:
    subset = [examples[i] for i in random.sample(range(len(examples)), numSamples)]

    # Train a tree without pruning
    trainedNodeNoPruning = ID3(subset, 0)
    accuraciesNoPruning.append(test(trainedNodeNoPruning, [random.choice(testDataset) for _ in range(100)]))

    # Train a tree with pruning
    trainedNodePruning = ID3(subset, 0)
    trainedNodePruning = prune(trainedNodePruning, testDataset)
    accuraciesPruning.append(test(trainedNodePruning, [random.choice(testDataset) for _ in range(100)]))

  print(accuraciesNoPruning)
  print(accuraciesPruning)
  print(numberTrainingSamples)

  # Plot the graphs
  plt.figure(figsize=(10, 6))
  plt.plot(numberTrainingSamples, accuraciesNoPruning, label="No Pruning")
  plt.plot(numberTrainingSamples, accuraciesPruning, label="Pruning")
  plt.xlabel('Number of Training Samples')
  plt.ylabel('Accuracy')
  plt.title('Training Samples Size Vs Accuracy')
  plt.legend()
  plt.show()

if __name__ == "__main__":
  examples = parse("house_votes_84.data")
  examples = impute_missing_value(examples)
  # examples = utility.removeMissingValues(examples)

  generateTrainingGraph(examples)
