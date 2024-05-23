from node import Node
from parse import parse
from collections import Counter
from math import log2

###################################################################
############################# HELPERS #############################
###################################################################

def attrs_vals(examples: list[dict]) -> dict[str:set]:
  """Returns a dictionary of attribute:values pairs

  Args:
      examples (list[dict]): An array of examples. Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class"

  Returns:
      dict[str:set]: A dictionary of attribute:values pairs
  """
  attributes = [attribute for attribute in examples[0] if attribute != 'Class']
  attribute_values = {attribute:set() for attribute in attributes}
  for example in examples:
    for attribute, value in example.items():
      if attribute != 'Class' and value != '?':
        attribute_values[attribute].add(value)
  return attribute_values

def are_all_same(examples: list[dict]) -> str|None:
  """Returns the class of a dataset if each example has the same class; otherwise, it returns nothing

  Args:
      examples (list[dict]): An array of examples. Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class"

  Returns:
      str|None: Class or None
  """
  first_class = examples[0]['Class']
  for example in examples:
    if example['Class'] != first_class:
      return None
  return first_class

def min_entropy(examples: list[dict], attributes: dict[str:set]) -> tuple[str,float]:
  """Returns a tuple of (attribute, entropy) for the attribute with the highest information gain

  Args:
      examples (list[dict]): An array of examples. Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class"
      attributes (dict[float:set]): A dictionary of attribute:values pairs

  Returns:
      tuple[str, float]: tuple of (attribute, info gain of attribute)
  """
  def entopry(attribute: str) -> float:
    value_to_count = {value:[] for value in attributes[attribute]}

    # initializing probabilities of value
    for example in examples:
      # TODO: handling for ?
      value_to_count[example[attribute]].append(example)

    # for each value of attribute, finding starting_entropy of data_value ⊆ data
    starting_entropy = 0
    for value_examples in value_to_count.values():
      # value_examples = [{Class: 1, Weather: Cold}, {Class: 0, Weather: Cold}]
      value_count = len(value_examples)
      prob_val = value_count/len(examples)
      class_to_count = {}
      # initializing class count for each value
      for ex in value_examples:
        # ex = {Class: 1, Weather: Cold}
        ex_class = ex['Class']
        if class_to_count.get(ex_class, False):
          class_to_count[ex_class] += 1
        else:
          class_to_count[ex_class] = 1
      
      val_h = 0
      for class_count in class_to_count.values():
        prob_class_in_value = class_count/value_count
        val_h += -1*(prob_class_in_value) * log2(prob_class_in_value)
      starting_entropy += prob_val * val_h

    return starting_entropy
  
  attr_entropy_list = []
  for attribute in attributes:
    attr_entropy_list.append((attribute, entopry(attribute)))
  
  return min(attr_entropy_list, key=lambda x: x[1])

def most_common_class(examples: list[dict], target: str='Class') -> str:
  """Returns the most common class of the dataset

  Args:
      examples (list[dict]): An array of examples. Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class"
      target (str, optional): Defaults to 'Class'.

  Returns:
      str: The most common class
  """
  classes: Counter = Counter([example[target] for example in examples])
  return classes.most_common(1)[0][0]

def most_common_class_of_node(node: Node) -> str:
  """Returns the most common class a node returns

  Args:
      node (Node): A node

  Returns:
      str: A class
  """
  class_to_count = {}

  def recurse(tree: Node=node) -> None:
    # base case 
    if not tree.children:
      class_ = tree.label
      if not class_to_count.get(class_, False):
        class_to_count[class_] = 1
      else:
        class_to_count[class_] += 1
      return
    for attribute, subtree in tree.children.items():
      if attribute != 'default':
        recurse(subtree)

  recurse()
  
  return max(class_to_count, key=lambda x: class_to_count[x])

def most_common_attr_val_for_class(examples: list[dict], attribute: str, values: set[str], target_class: str) -> str:
  """Returns the most common value of an attribute with a specific class

  Args:
      examples (list[dict]): An array of examples. Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class"
      attribute (str): A specific attribute
      values (list[str]): A list of values for a specific attribute
      target_class (str): The class of the missing attribute

  Returns:
      str: The most common value
  """
  values_to_count = {value:0 for value in values[attribute]}
  for example in examples:
    value = example[attribute]
    if example['Class'] != target_class or value == '?':
      continue
    values_to_count[value] += 1

  return max(values_to_count, key=lambda x: values_to_count[x])

def clean_examples(examples: list[dict], attrs_vals: dict[str:set]) -> None:
  """Modifies the dataset by replacing every '?' in each of its examples with the most common value of the corresponding attribute

  Args:
      examples (list[dict]): A dataset
  """
  for example in examples:
    class_ = example['Class']
    for attribute, value in example.items():
      if value != '?':
        continue
      example[attribute] = most_common_attr_val_for_class(examples, attribute, attrs_vals, class_)

###################################################################
############################ FUNCTIONS ############################
###################################################################

def ID3(examples: list[dict], default: str=None) -> Node:
  """Takes in an array of examples, and returns a tree (an instance of Node)
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"

  Args:
      examples (list[dict]): An array of examples. Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
      default (str): None

  Returns:
      Node: A tree (an instance of Node) trained on the examples
  """
  # initialize attributes
  attributes = attrs_vals(examples)

  # fill in '?'s with most common attribute value of its class
  clean_examples(examples, attributes)

  # recursive ID3 function
  def recurse(data: list[dict]=examples, attributes: dict[str:set]=attributes, target: str='Class') -> Node:
    # base cases
    # !None = all_same
    if are_all_same(data) != None:
      return Node(are_all_same(data))
    node = Node(most_common_class(data))
    if not len(attributes):
      return node
    
    # recursive part
    best_attribute = min_entropy(data, attributes)[0] # [0] is the attribute (str)
    node.label = best_attribute

    # attach a default node in case we haven't seen attribute value or it's a '?'
    if default:
      node.children['default'] = default
    else:
      node.children['default'] = most_common_class(data)

    # attributes[best_attribute] = {val_1 ('0'), val_2 ('1'),...}
    for value in attributes[best_attribute]:
      value_data = [d for d in data if d[best_attribute] == value] # list[dict]
      if not value_data:
        value_node = Node(most_common_class(data))
        # TODO: change node data structure
        node.children[value] = value_node
        continue

      value_attributes = {attribute:values.copy() for attribute, values in attributes.items() if attribute != best_attribute}
      value_node = recurse(value_data, value_attributes, 'Class')
      node.children[value] = value_node

    return node
    
  return recurse()

def prune(node: Node, examples: list[dict]) -> None:
  """Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.

  Args:
      node (Node): a trained tree
      examples (list[dict]): a validation set of examples
  """
  # reduced error pruning
  def recurse(tree: Node=node) -> None:
    accuracy = test(node, examples)
    for value, subtree in tree.children.items():
      # base cases
      if value == 'default':
        continue
      if not subtree.children:
        continue

      # change node to most common class
      tree.children[value] = Node(most_common_class_of_node(subtree))
      new_accuracy = test(node, examples)

      # put node back if worse, then recurse on node
      if new_accuracy < accuracy:
        tree.children[value] = subtree
        recurse(subtree)
      # if we keep pruned node, continue to next child

  recurse()

def test(node: Node, examples: list[dict], clean: bool=False) -> float:
  """Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).

  Args:
      node (Node): a trained tree
      examples (list[dict]): a test set of examples

  Returns:
      float: the accuracy (fraction of examples the tree classifies correctly)
  """
  if clean:
    values = attrs_vals(examples)
    clean_examples(examples, values)
    
  correct = 0
  total = len(examples)

  for example in examples:
    preditction = evaluate(node, example)
    actual = example['Class']
    if preditction == actual:
      correct += 1

  return correct/total

def evaluate(node: Node, example: dict[str:str]) -> str:
  """Takes in a tree and one example.  Returns the Class value that the tree
    assigns to the example.

  Args:
      node (Node): a tree
      example (dict): one example

  Returns:
      int: the Class value that the tree assigns to the example
  """
  # assumes an example wouldn't have any '?'s
  def recurse(tree: Node=node) -> str:
    if not tree.children:
      return tree.label
    attribute = tree.label
    value = example[attribute]
    next_tree = tree.children.get(value, False)
    # if missing value for attribute
    if not next_tree or value == '?':
      return tree.children['default']
    else:
      return recurse(next_tree)

  return recurse()

###################################################################
############################## MAIN! ##############################
###################################################################

from pprint import pprint
# for testing
if __name__ == '__main__':
  examples = parse("house_votes_84.data")
  pprint(examples[:4])
  # examples = parse("candy.data")
  # examples = parse("tennis.data")
  # examples = parse('restaurant.data')
  vals = attrs_vals(examples)

  training_data = examples[:len(examples)//2]
  test_data = examples[len(examples)//2:]
  # clean_examples(test_data, vals)
  tree = ID3(training_data)

  print(round(test(tree, test_data, clean=True), 3), "accuracy on test")

  prune(tree, test_data)

  print("-"*15, "After pruning", "-"*15)
  print(round(test(tree, test_data), 3), "accuracy on test")