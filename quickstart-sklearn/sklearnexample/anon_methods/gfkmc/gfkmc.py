import numpy as np
import pandas as pd
from scipy.stats import rankdata
import random
from ..tree.gentree import read_tree, print_tree
import math
import timeit
from typing import List
import itertools


class Cluster:
    """cluster class"""
    # centroid register index
    centroid: int
    # register indices
    r_indices: List[int]

    def __init__(self, centroid, r_indices):
        self.centroid = centroid
        self.r_indices = r_indices

    def __str__(self):
        return f'centroid={self.centroid}, indices={self.r_indices}'

    @property
    def size(self):
        return len(self.r_indices)

class Group:
    """group class"""
    # register indices
    r_indices: List[int]

    def __init__(self, r_indices):
        self.r_indices = r_indices

    @property
    def size(self):
        return len(self.r_indices)

class GFKMCTable:
    """gfkmc table and methods class"""
    # table df
    df: pd.DataFrame
    # original table df
    df_orig: pd.DataFrame
    # numerical attributes
    numericals: List[str]
    # categorical QI attributes
    categoricals: List[str]
    # categorical sensitive attributes
    sensitives: List[str]
    # quasi-identifiers
    QIs: List[str]
    # attribute tree
    att_tree: dict
    # list of clusters
    clusters: List[Cluster]
    # list of remaining clusters formed in grouping and adjustment phases
    remaining_clusters: List[Cluster]
    # remaning groups
    remaining_groups: List[Group]
    # outlier groups indices
    outlier_groups: List[int]
    # outlier groups total registers count
    outlier_count: int
    # non outlier groups indices
    non_outlier_groups: List[int]
    # non outlier groups total registers count
    non_outlier_count: int
    # weight score of each register
    weight_scores: pd.Series


    def __init__(self, df, df_orig, numericals, categoricals, sensitives, att_tree):
        self.df = df.copy()
        self.df_orig = df_orig
        self.numericals = numericals
        self.categoricals = categoricals
        self.sensitives = sensitives
        self.QIs = numericals + categoricals
        self.categoricals_all = categoricals + sensitives
        self.att_tree = att_tree
        self.remaining_groups = []
        self.clusters = []
        self.remaining_clusters = []
        self.outlier_groups = []
        self.outlier_count = 0
        self.non_outlier_groups = []
        self.non_outlier_count = 0
        self.weight_scores = pd.Series()

        # caching
        self._table_max = None
        self._table_min = None

    def generalize_to_parent(self):
        # generalize categorical attributes to parent value if parent is not '*'
        for cat in self.categoricals:
            self.df.loc[self.df.index, cat] = self.df[cat].apply(lambda value: self.att_tree[cat][value].parent[0].value if self.att_tree[cat][value].parent[0].value != '*' else self.att_tree[cat][value].value)


    def initial_clustering_phase(self, k: int, generalize=True):
        if generalize:
            self.generalize_to_parent()

        # group registers with same QIs and form a cluster for every group that has at least k members
        grouped = self.df.groupby(self.QIs)
        group_indices = grouped.indices
        sorted_groups = grouped.size().sort_values(ascending=False)
        group_counts = [(group_indices[row], count) for row, count in zip(sorted_groups.index, sorted_groups)]

        remaining_groups_indices = []
        for i, (indices, count) in enumerate(group_counts):
            if count < k:
                remaining_groups_indices = group_counts[i:]
                break
            centroid = indices[0]
            self.clusters.append(Cluster(centroid, indices))

        # return the remaining groups that have less than k members
        return remaining_groups_indices


    def n_score(self, r, numerical_average):
        # print(np.array(r) - np.array(numerical_average))
        return np.linalg.norm(np.array(r) - numerical_average, ord=1)
        # return np.sum(np.abs(np.array(r) - np.array(numerical_average)))

    # attribute value frequency (avf) score
    def c_score(self, r, value_counts_dict):
        n = len(self.categoricals)
        result = 0
        for column in value_counts_dict:
            result += value_counts_dict[column][r[column]]
        return 1/n * result

    def ar_score(self, n_scores, c_scores):
        # print(rankdata(n_scores, method='ordinal'), len(n_scores) + 1 - rankdata(n_scores, method='ordinal'))
        return 1/2 * ((len(n_scores) + 1 - rankdata(n_scores, method='ordinal')) + rankdata(c_scores, method='ordinal'))


    def weighting_phase(self, beta, remaining_groups_indices):
        # compute the weight score for all registers of the remaining groups
        remaining_indices = np.array([])
        for indices, count in remaining_groups_indices:
            remaining_indices = np.append(remaining_indices, indices).astype(int)

        local_df = self.df.loc[remaining_indices]
        if self.numericals:
            numerical_average = np.average(np.array(local_df[self.numericals]), axis=0)
        value_counts_dict = {col: local_df[col].value_counts() for col in self.categoricals}

        scores = None
        for indices, count in remaining_groups_indices:
            r = local_df.loc[indices[0]]
            if self.numericals:
                n_score_value = self.n_score(r[self.numericals], numerical_average)
            else:
                n_score_value = 0
            c_score_value = self.c_score(r[self.categoricals], value_counts_dict)
            local_scores = np.array([(n_score_value, c_score_value)] * count)

            if scores is None:
                scores = local_scores
            else:
                scores = np.vstack((scores, local_scores))


        ar_scores = self.ar_score(scores[:, 0], scores[:, 1])
        weight_scores = local_df.shape[0] - ar_scores
        self.weight_scores = pd.Series(dict(zip(remaining_indices, weight_scores)))

        # store remaining groups in a list of Groups for convenience
        for indices, count in remaining_groups_indices:
            self.remaining_groups.append(Group(indices.tolist()))


        # separate outliers and non_outliers according to beta value
        # if beta is 0, then there are no outliers
        if beta == 0:
            for i, group in enumerate(self.remaining_groups):
                self.non_outlier_groups.append(i)
                self.non_outlier_count += len(group.r_indices)

        else:
            # sort the weight scores in decreasing order
            weight_scores = sorted(weight_scores, reverse=True)

            # get the outlier threshold weight score according to beta
            outlier_weight_value = weight_scores[beta-1]

            # for each group, if weight score is greater or equal to the
            # threshold, add group as outlier, else add group as non_outlier
            for i, group in enumerate(self.remaining_groups):
                r_index = group.r_indices[0]
                weight_value = self.weight_scores[r_index]
                if weight_value >= outlier_weight_value:
                    self.outlier_groups.append(i)
                    self.outlier_count += len(group.r_indices)
                else:
                    self.non_outlier_groups.append(i)
                    self.non_outlier_count += len(group.r_indices)


        # print(len(self.outlier_groups) + len(self.non_outlier_groups), len(remaining_groups_indices))


    def distance(self, r_index1, r_index2):
        def dn(r1, r2):
            if self._table_max is None:
                self._table_max = np.max(self.df_orig[self.numericals], axis=0)
            if self._table_min is None:
                self._table_min = np.min(self.df_orig[self.numericals], axis=0)
            return np.abs(r1 - r2) / (self._table_max - self._table_min)

        def dc(r1, r2):
            results = np.zeros(len(self.categoricals))
            for i, c in enumerate(self.categoricals):
                root = self.att_tree[c]['*']
                # print([c], [r1[c]])
                parent_leafs = root.lca(self.att_tree[c][r1[c]], self.att_tree[c][r2[c]]).leaf_num
                tree_leafs = root.leaf_num
                # print(root.lca(self.att_tree[c][r1[c]],self.att_tree[c][r2[c]]).value, parent_leafs)
                results[i] = parent_leafs / tree_leafs

            # print(np.where(r1 == r2, 0, results))
            return np.where(r1 == r2, 0, results)

        r1 = self.df.loc[r_index1, self.QIs]
        r2 = self.df.loc[r_index2, self.QIs]

        if np.array_equal(r1, r2):
            result = 0
        else:
            result = np.sum(dn(r1[self.numericals], r2[self.numericals])) + np.sum(dc(r1[self.categoricals], r2[self.categoricals]))
        # print(result)
        return result

    def weight_distance(self, r_index1, r_index2):
        return np.sqrt(self.weight_scores[r_index1]**2 + self.weight_scores[r_index2]**2) * self.distance(r_index1, r_index2)

    # information loss
    def il(self, cluster_indices):
        cn = self.df.loc[cluster_indices, self.numericals]
        cc = self.df.loc[cluster_indices, self.categoricals]

        il_n = np.sum((cn.max(axis=0) - cn.min(axis=0)) / len(self.numericals))
        il_c = 0
        for c in self.categoricals:
            root = self.att_tree[c]['*']
            c_values = cc[c].unique()
            lca = self.att_tree[c][c_values[0]]
            for c_value in c_values[1:]:
                # print([c], [c_value])
                lca = root.lca(lca, self.att_tree[c][c_value])
            h_lca = lca.height()
            h_tree_c = self.att_tree[c]['*'].height()
            il_c += h_lca / h_tree_c

        return len(cluster_indices) * (il_n + il_c)

    # weight information loss
    def wil(self, cluster_indices):
        # print(f'[cluster_indices]={cluster_indices}')
        # print(f'self.weight_scores[cluster_indices]={self.weight_scores[cluster_indices]}')
        # print(cluster_indices)
        return np.linalg.norm(self.weight_scores[cluster_indices], ord=2) * self.il(cluster_indices)


    def find_next_centroid(self, groups_indices, D):
        dists = np.zeros((len(groups_indices) * len(self.remaining_clusters), 2))
        i = 0
        # for each group
        for group_index in groups_indices:
            group = self.remaining_groups[group_index]
            indices = group.r_indices
            r_index = indices[0]

            # for each cluster, compute the weight distance of one register
            # of the group (as all registers of a group have the same QIs)
            # to the centroid of the previously formed clusters
            for cluster_index, cluster in enumerate(self.remaining_clusters):
                centroid = cluster.centroid
                dist = D[group_index][cluster_index]
                if (dist != -1):
                    dist_r = dist
                else:
                    dist_r = self.weight_distance(r_index, centroid)
                    D[group_index][cluster_index] = dist_r

                dists[i] = np.array((group_index, dist_r))
                i += 1

        # select the furthest group as next centroid
        next_centroid = int(dists[dists[:, 1].argmax(), 0])

        return next_centroid

    def find_min_wil_groups(self, cluster_index):
        current_cluster_indices = self.remaining_clusters[cluster_index].r_indices
        current_cluster_wil = self.wil(current_cluster_indices)
        groups_wil = np.zeros((len(self.non_outlier_groups), 2))

        for i, group_index in enumerate(self.non_outlier_groups):
            group = self.remaining_groups[group_index]
            wil_value = self.wil(np.append(current_cluster_indices, [group.r_indices[0]])) - current_cluster_wil
            groups_wil[i] = [group_index, wil_value]

        sorted_records = groups_wil[groups_wil[:, 1].argsort()][:, 0].astype(int)
        return sorted_records


    # method=['cluster_size', 'merge_entire_group']
    def grouping_phase(self, k, method='cluster_size'):
        # S = number of clusters ot form
        S = math.floor(self.non_outlier_count / k)

        # D = [group, cluster] distance matrix
        D = np.full((len(self.remaining_groups), S), -1)

        cluster_index = 0
        # print(self.non_outlier_count)

        # while the number of outlier registers is greater or equal to k
        while self.non_outlier_count >= k:
            # find next centroid group
            if (len(self.remaining_clusters) < 1):
                next_centroid = random.choice(self.non_outlier_groups)
            else:
                next_centroid = self.find_next_centroid(self.non_outlier_groups, D)

            # create a new cluster with the centroid group and remove from the non_outlier_groups
            group = self.remaining_groups[next_centroid]
            self.remaining_clusters.append(Cluster(group.r_indices[0], group.r_indices))
            self.non_outlier_count -= group.size
            self.non_outlier_groups.remove(next_centroid)

            print(f'centroid group={group.r_indices}')

            # compute the wil of all groups with the current cluster
            # and sort the groups in ascending order by wil value
            min_group_indices = self.find_min_wil_groups(cluster_index)
            cluster = self.remaining_clusters[cluster_index]
            for min_group_index in min_group_indices:
                n_registers_to_fill_cluster = k - cluster.size
                if n_registers_to_fill_cluster <= 0:
                    break

                group = self.remaining_groups[min_group_index]

                if method == 'merge_entire_group':
                    cluster.r_indices.extend(group.r_indices)
                    self.non_outlier_groups.remove(min_group_index)
                    self.non_outlier_count -= group.size

                # fill cluster with enough registers to reach k registers
                elif method == 'cluster_size':
                    if group.size <= n_registers_to_fill_cluster:
                        cluster.r_indices.extend(group.r_indices)
                        self.non_outlier_groups.remove(min_group_index)
                        self.non_outlier_count -= group.size
                    else:
                        cluster.r_indices.extend(group.r_indices[:n_registers_to_fill_cluster])
                        group.r_indices = group.r_indices[n_registers_to_fill_cluster:]
                        self.non_outlier_count -= n_registers_to_fill_cluster

            cluster_index += 1

        # print(f"non_outlier_count: {self.non_outlier_count}, outlier_count: {self.outlier_count}")
        # print(f"non_outlier_groups: {self.non_outlier_groups}, outlier_groups: {self.outlier_groups}")
        # for group in self.non_outlier_groups:
        #     print(f"remaining_groups_weight[{group}].r_indices: {self.remaining_groups[group].r_indices}")



    def find_min_wil_cluster(self, group_index):
        group = self.remaining_groups[group_index]

        min_wil_cluster_index = 0
        current_cluster_indices = self.remaining_clusters[min_wil_cluster_index].r_indices
        min_wil = self.wil(np.append(current_cluster_indices, group.r_indices)) - self.wil(current_cluster_indices)
        for i, cluster in enumerate(self.remaining_clusters[1:], start=1):
            current_cluster_indices = cluster.r_indices
            new_min_wil = self.wil(np.append(current_cluster_indices, group.r_indices)) - self.wil(current_cluster_indices)
            if new_min_wil < min_wil:
                min_wil = new_min_wil
                min_wil_cluster_index = i

        return min_wil_cluster_index

    def adjustment_phase(self):
        # while remaining non outlier registers is not 0, find the cluster with min wil to add the group
        while self.non_outlier_count != 0:
            group_index = random.choice(self.non_outlier_groups)

            min_wil_cluster_index = self.find_min_wil_cluster(group_index)
            group = self.remaining_groups[group_index]
            self.remaining_clusters[min_wil_cluster_index].r_indices.extend(group.r_indices)
            self.non_outlier_groups.remove(group_index)
            self.non_outlier_count -= group.size

        # while outlier registers is not 0, find the cluster with min wil to add the group
        while (self.outlier_count != 0):
            group_index = random.choice(self.outlier_groups)

            min_wil_cluster_index = self.find_min_wil_cluster(group_index)
            group = self.remaining_groups[group_index]
            self.remaining_clusters[min_wil_cluster_index].r_indices.extend(group.r_indices)
            self.outlier_groups.remove(group_index)
            self.outlier_count -= group.size

        self.clusters.extend(self.remaining_clusters)


    # method=['most_common_register', 'most_common_values', 'cluster_centroid']
    def cluster_generalization(self, method):
        df = self.df.copy()
        for cluster in self.clusters:
            cluster_df = self.df.loc[cluster.r_indices, self.QIs]
            if method == 'cluster_centroid':
                # print(cluster_df.loc[cluster.centroid].values)
                row = cluster_df.loc[cluster.centroid].values
            elif method == 'most_common_values':
                cols = cluster_df.columns
                row = []
                for col in cols:
                    row.append(cluster_df[col].value_counts(sort=True).keys()[0])
            elif method == 'most_common_register':
                row = cluster_df.value_counts(sort=True).keys()[0]
            df.loc[cluster.r_indices, self.QIs] = row

        return df



    def print_clusters(self):
        for cluster in self.clusters:
            print(cluster)
            print(f'{self.df.loc[cluster.r_indices]}')
        print()


    ###### metrics

    def intra_diss(self, cluster: Cluster):
        return  1/cluster.size**2 * np.sum([self.distance(i, j) for i, j in itertools.combinations(cluster.r_indices, 2)])

    def inter_diss(self, cluster1: Cluster, cluster2: Cluster):
        return 1/(cluster1.size*cluster2.size) * np.sum([self.distance(i, j) if j != i else 0 for i in cluster1.r_indices for j in cluster2.r_indices])

    def ncp(self, df_anon: pd.DataFrame):
        if self._table_max is None:
            self._table_max = np.max(self.df_orig[self.numericals], axis=0)
        if self._table_min is None:
            self._table_min = np.min(self.df_orig[self.numericals], axis=0)
        # print(self._table_max, self._table_min)

        table_cat_unique_count = dict()
        for cat in self.categoricals:
            table_cat_unique_count[cat] = self.att_tree[cat]['*'].leaf_num

        n_QIs = len(self.QIs)
        ncp = 0
        for cluster in self.clusters:
            cluster_df_num = self.df_orig.loc[cluster.r_indices, self.numericals]
            cluster_max = np.max(cluster_df_num, axis=0)
            cluster_min = np.min(cluster_df_num, axis=0)
            # print(cluster_max.to_numpy(), cluster_min.to_numpy(), self._table_max, self._table_min)
            # print(np.sum((cluster_max - cluster_min) / (self._table_max - self._table_min)))
            # print((cluster_max - cluster_min) / (self._table_max - self._table_min))
            local_ncp = np.sum((cluster_max - cluster_min) / (self._table_max - self._table_min))


            for cat in self.categoricals:
                cluster_cat_values = df_anon.loc[cluster.r_indices, cat].unique().tolist()
                if len(cluster_cat_values) > 1:
                    root = self.att_tree[cat]['*']
                    c_values = cluster_cat_values
                    lca = self.att_tree[cat][c_values[0]]
                    for c_value in c_values[1:]:
                        lca = root.lca(lca, self.att_tree[cat][c_value])
                else:
                    lca = self.att_tree[cat][cluster_cat_values[0]]
                # print(cluster_cat_values, lca.leaf_num)
                # print(lca.parent[0].leaf_num)

                if lca.leaf_num == 1:
                    # ncp += 0
                    continue

                # print(lca.leaf_num / table_cat_unique_count[cat])
                # print(f'{cat} {lca.leaf_list}   {lca.leaf_num} / {table_cat_unique_count[cat]}')
                local_ncp += lca.leaf_num / table_cat_unique_count[cat]

            local_ncp = local_ncp / n_QIs
            ncp += cluster.size * local_ncp
            # print(ncp)


        n_registers = df_anon.shape[0]
        # print(n_QIs, n_registers)
        # print(ncp, n_QIs * n_registers)
        ncp /= (n_registers)

        return ncp




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
    att_tree = read_tree('./tree/adult/', att_names)
    print_tree(att_tree["native-country"]['*'])

    # print(new_df.value_counts())
    # np.savetxt("./b.txt", new_df.value_counts().to_numpy(), fmt="%d")

    # table = GFKMCTable(new_df, df_orig, numericals, categoricals, sensitives, att_tree)

    # local_start = timeit.default_timer()
    # remaining_groups_indices = table.initial_clustering_phase(k)
    # stop = timeit.default_timer()
    # execution_time = stop - local_start
    # print(f"initial_clustering_phase execution time: {execution_time}")


    # local_start = timeit.default_timer()
    # beta = int(len(remaining_groups_indices) * 0.05)
    # table.weighting_phase(beta, remaining_groups_indices)
    # stop = timeit.default_timer()
    # execution_time = stop - local_start
    # print(f"weighting_phase execution time: {execution_time}")

    # local_start = timeit.default_timer()
    # table.grouping_phase(k)
    # stop = timeit.default_timer()
    # execution_time = stop - local_start
    # print(f"grouping_phase execution time: {execution_time}")

    # local_start = timeit.default_timer()
    # table.adjustment_phase()
    # stop = timeit.default_timer()
    # execution_time = stop - local_start
    # print(f"adjustment_phase execution time: {execution_time}")


    # table.cluster_generalization('most_common_register')


    # table.print_clusters()
    # print(table.remaining_clusters)
    # df = table.df.copy()
    # df['cluster'] = -1
    # for i, cluster in enumerate(table.clusters):
    #     df.loc[cluster.r_indices, 'cluster'] = i
    # print(df)


    # for i, cluster in enumerate(table.clusters):
    #     intra_diss = table.intra_diss(cluster)
    #     print(f'table.intra_diss(table.clusters[{i}]) = {intra_diss}')

    # for pair in itertools.combinations(range(len(table.clusters)), 2):
    #     inter_diss = table.inter_diss(table.clusters[pair[0]], table.clusters[pair[1]])
    #     print(f'table.inter_diss(table.clusters[{pair[0]}], table.clusters[{pair[1]}]) = {inter_diss}')



if __name__ == "__main__":
    main()
