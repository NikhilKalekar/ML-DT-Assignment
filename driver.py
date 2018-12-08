from DecisionTree import *
import pandas as pd
from sklearn import model_selection

header = ['A1', 'A2', 'A3', 'A4', 'A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15','Class']
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data', header=None, names=['A1', 'A2', 'A3', 'A4', 'A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15','Class'])
lst = df.values.tolist()
t = build_tree(lst, header)
print_tree(t)

print("********** Leaf nodes ****************")
leaves = getLeafNodes(t)
for leaf in leaves:
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))
print("********** Non-leaf nodes ****************")
innerNodes = getInnerNodes(t)
for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))

trainDF, testDF = model_selection.train_test_split(df, test_size=0.2)
train = trainDF.values.tolist()
test = testDF.values.tolist()

t = build_tree(train, header)
print("*************Tree before pruning*******")
print_tree(t)
acc = computeAccuracy(test, t)
print("Accuracy on test = " + str(acc))
acc1 = computeAccuracy(train, t)
print("Train Accuracy", str(acc1))
## TODO: You have to decide on a pruning strategy
#t_pruned = prune_tree(t, [1024, 1025, 8193, 8192])
#t_pruned = prune_tree(t, [1579,791,395,53,107,26,5])
t_pruned = prune_tree(t, [1579,791,219,26,54,12])
print("*************Tree after pruning*******")
print_tree(t_pruned)
acc = computeAccuracy(test, t)
print("Accuracy on test = " + str(acc))
