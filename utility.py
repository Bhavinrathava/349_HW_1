import math

def findInformationGain(examples, attirbute):
    #Do Something
    pass

def findBestAttribute(examples, attributeList):
    pass

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



 