class Node:
  def __init__(self, label = None, attribute = None):
    self.label = label # If Label is not null -> leaf node
    self.attribute = attribute
    self.children = {} # Attribute[val1] : Node, Attribute[val2] : Node , ....
    
    
	# you may want to add additional fields here...