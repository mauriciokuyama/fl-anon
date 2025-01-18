# adapted from: https://github.com/kaylode/k-anonymity/
import os
import pandas as pd

class GenTree(object):

    """Class for Generalization hierarchies (Taxonomy Tree).
    Store tree node in instances.
    self.value: node value
    self.level: tree level (top is 0)
    self.leaf_num: number of leaf node covered
    self.parent: ancestor node list
    self.child: direct successor node list
    self.cover: all nodes covered by current node
    """

    def __str__(self):
        return f"Node(value={self.value}, level={self.level}, leaf_num={self.leaf_num}, parent={[parent.value for parent in self.parent]}, child={[child.value for child in self.child]}, cover={[node.value for node in self.cover.values()]}, isleaf={self.isleaf}, leaf_list={self.leaf_list})"

    def __init__(self, value=None, parent=None, isleaf=False):
        self.value = ''
        self.level = 0
        self.leaf_num = 0
        self.parent = []
        self.child = []
        self.cover = {}
        self.isleaf = isleaf
        self.leaf_list = []
        self.code = 0
        self._lca_cache = {}

        if value is not None:
            self.value = value
            self.cover[value] = self
        if parent is not None:
            self.parent = parent.parent[:]
            self.parent.insert(0, parent)
            parent.child.append(self)
            self.level = parent.level + 1
            for t in self.parent:
                t.cover[self.value] = self
                if isleaf:
                    t.leaf_num += 1
                    t.leaf_list.append(self.value)

            if parent.code == 0:
                self.code = (parent.code + len(parent.child))*10
            else:
                self.code = parent.code*10 + len(parent.child)

    def node(self, value):
        """Search tree with value, return GenTree node.
        return point to that node, or None if not exists
        """
        try:
            return self.cover[value]
        except KeyError:
            return None

    def __len__(self):
        """
        return number of leaf node covered by current node
        """
        return self.leaf_num


    def height(self):
        if not self.child:
            return 0
        else:
            return 1 + max(child.height() for child in self.child)

    def lca(self, node1, node2):
        def find_path(node, target, path):
            if node is None:
                return False
            path.append(node)
            if node == target:
                return True
            if any(find_path(child, target, path) for child in node.child):
                return True
            path.pop()
            return False

        path1 = []
        path2 = []

        if (self, node1, node2) in self._lca_cache or (self, node2, node1) in self._lca_cache:
            return self._lca_cache[(self, node1, node2)]

        find_path(self, node1, path1)
        find_path(self, node2, path2)

        if not path1 or not path2:
            return None

        lca_node = None
        for n1, n2 in zip(path1, path2):
            if n1 == n2:
                lca_node = n1
            else:
                break

        self._lca_cache[(self, node1, node2)] = lca_node
        self._lca_cache[(self, node2, node1)] = lca_node
        return lca_node



def read_tree_file(path, treename):
    att_tree = {}
    prefix = os.path.join(path, 'hierarchy_')
    postfix = ".csv"
    with open(prefix + treename + postfix) as treefile:
        att_tree['*'] = GenTree('*')
        for line in treefile:
            # delete \n
            if len(line) <= 1:
                break
            line = line.strip()
            temp = line.split(';')
            # copy temp
            temp.reverse()
            for i, t in enumerate(temp):
                isleaf = False
                if i == len(temp) - 1:
                    isleaf = True

                # try and except is more efficient than 'in'
                try:
                    att_tree[t]
                except KeyError:
                    att_tree[t] = GenTree(t, att_tree[temp[i - 1]], isleaf)
    return att_tree

def read_tree(path, att_names):
    att_trees = {}
    for att_name in att_names:
        att_trees[att_name] = read_tree_file(path, att_name)
    return att_trees

def print_tree(node: GenTree, depth=0):
    if node is None:
        return

    print("  " * depth + f"- {node.value} ({node.code})")
    for child in node.child:
        print_tree(child, depth + 1)


def main():
    data = {
        'age': [2, 3, 2, 3, 1, 2, 3, 3, 1, 2, 2],
        'workclass': ['State-gov', 'Self-emp-inc', 'Private', 'Private', 'Private', 'Private', 'Private', 'Self-emp-inc', 'Private', 'Private', 'Private'],
        'education-num': [13, 13, 9, 7, 13, 14, 5, 9, 14, 13, 10],
        'marital-status': ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-civ-spouse', 'Married-civ-spouse', 'Married-civ-spouse', 'Separated', 'Married-civ-spouse', 'Never-married', 'Married-civ-spouse', 'Married-civ-spouse'],
        'occupation': ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Handlers-cleaners', 'Prof-specialty', 'Exec-managerial', 'Other-service', 'Exec-managerial', 'Prof-specialty', 'Exec-managerial', 'Exec-managerial'],
        'race': ['White', 'White', 'White', 'Black', 'Black', 'White', 'Black', 'White', 'White', 'White', 'Black'],
        'sex': ['Male', 'Male', 'Male', 'Male', 'Female', 'Female', 'Female', 'Male', 'Female', 'Male', 'Male'],
        'native-country': ['United-States', 'United-States', 'United-States', 'United-States', 'Cuba', 'United-States', 'Jamaica', 'United-States', 'United-States', 'United-States', 'United-States'],
        'salary-class': ['>50K', '>50K', '>50K', '>50K', '<=50K', '<=50K', '<=50K', '>50K', '>50K', '<=50K', '<=50K']
    }
    df = pd.DataFrame(data)
    df_orig = df.copy()
    k = 3
    new_df = df

    # df = pd.read_csv("datasets/adult_num_anon.csv")
    # QIs = ['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'race', 'sex', 'native-country']
    # k = 100
    # new_df = df[QIs]

    categoricals = ['workclass', 'marital-status', 'occupation', 'native-country']
    numericals = ['age', 'education-num']
    sensitives = ['salary-class', 'race', 'sex']

    att_names = df[categoricals + sensitives].columns
    print(att_names)
    att_tree = read_tree('../tree/adult/', att_names)
    for att in att_names:
        print_tree(att_tree[att]['*'])

if __name__ == "__main__":
    main()
