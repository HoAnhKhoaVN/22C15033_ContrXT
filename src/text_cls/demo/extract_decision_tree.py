from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from sklearn import tree
from pydot import graph_from_dot_data

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)

if __name__ == "__main__":
    # Load sample dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    print(f'y: {y}')

    feature_names = iris.feature_names
    print(f'feature_names: {feature_names}')

    # Train a decision tree classifier
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X, y)

    # Save the tree
    graph_str = tree.export_graphviz( 
            decision_tree=clf,
            class_names= ['0', '1', '2'],
            feature_names= feature_names,
            filled= True
        )
    fname = "example_decision_tree.png"
    (graph, )  = graph_from_dot_data(graph_str)
    graph.write_png(fname)

    # Extract and print the rules
    tree_to_code(clf, feature_names)
