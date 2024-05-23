class Node:
  def __init__(self, label:int=None, children:dict=None):
    self.label = label
    if children == None:
      self.children = dict()
    else:
      self.children = children