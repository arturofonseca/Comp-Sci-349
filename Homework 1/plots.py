import ID3, parse, random

def testPruningOnHouseData(inFile):
  data = parse.parse(inFile)
  withPruning = []
  withoutPruning = []
  for j in range(10, 301):
    pruning = []
    noPruning = []
    for i in range(10):
        random.shuffle(data)
        train = data[:j]
        valid = data[j:3*len(data)//4]
        test = data[3*len(data)//4:]
    
        tree = ID3.ID3(train, 'democrat')
        acc = ID3.test(tree, train)
        # print("training accuracy: ",acc)
        acc = ID3.test(tree, valid)
        # print("validation accuracy: ",acc)
        acc = ID3.test(tree, test)
        # print("test accuracy: ",acc)
    
        ID3.prune(tree, valid)
        acc = ID3.test(tree, train)
        # print("pruned tree train accuracy: ",acc)
        acc = ID3.test(tree, valid)
        # print("pruned tree validation accuracy: ",acc)
        acc = ID3.test(tree, test)
        # print("pruned tree test accuracy: ",acc)
        pruning.append(acc)
        tree = ID3.ID3(train+valid, 'democrat')
        acc = ID3.test(tree, test)
        # print("no pruning test accuracy: ",acc)
        noPruning.append(acc)
    avgPruning = round(sum(pruning)/len(pruning), 4)
    avgNoPruning = round(sum(noPruning)/len(noPruning), 4)
    withPruning.append(avgPruning)
    withoutPruning.append(avgNoPruning)

  print(*withoutPruning, sep='\n')
  print('*'*50)
  print(*withPruning, sep='\n')

if __name__ == '__main__':
  testPruningOnHouseData('house_votes_84.data')