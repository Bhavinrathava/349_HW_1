import math

def findEntropyGain(examples, attribute):
    # Find the entropy for the entirety of examples dataset
    originalEntropy = calculateEntropy(examples)
    afterEntropy = 0
    # Find unique values for attribute
    uniqueVals = getUniqueValuesForAttribute(examples, attribute)

    # for val in unique values:
    for val in uniqueVals:
        # Find Subset with attribute = val
        subset = getDataWithAttValue(examples, attribute, val)
    
        # Find the entropy for this subset
        # weight this with len(sub) / len(data)
        #add to the entropy after element
        afterEntropy += calculateEntropy(subset) *(len(subset) / len(examples))

    # Information gain = old - new entropy
    return originalEntropy - afterEntropy
        
def findBestAttribute(examples, attributes):
    
    infGainMap = {attr : findEntropyGain(examples, attr) for attr in attributes}

    res = ""
    infG = 0
    for atr in list(infGainMap.keys()):
        if(infGainMap[atr] > infG):
            infG = infGainMap[atr]
            res = atr
    
    return res

def getDataWithAttValue(examples, attribute, value):
    result = []

    for ex in examples:
        if(ex[attribute] == value):
            result.append(ex)
    
    return result

def getUniqueValuesForAttribute(examples, attribute):
    uniqueValues = []
    for ex in examples:
        if(ex[attribute] not in uniqueValues):
            uniqueValues.append(ex[attribute])
    
    return uniqueValues


#This function assesses the entropy on the passed dataset
def calculateEntropy(examples):    
    TARGET = "Class"
    counts = {}
    
    #Find the Number of total classes and get their counts
    for example in examples:
        if(example[TARGET] in counts):
            counts[example[TARGET]] += 1
        else:
            counts[example[TARGET]] = 1

    
    #Calculate the entropy 
    total = len(examples)
    entropy = 0

    for count in len(counts.keys()):
        p = (counts[count] / total)
        entropy -= p * math.log2(p)

    return entropy



 