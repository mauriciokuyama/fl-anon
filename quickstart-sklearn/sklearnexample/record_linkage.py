import pandas as pd
import numpy as np
import timeit

def record_linkage(df_orig, df_anon, att_tree, numericals, categoricals, num_intervals=False):
    if numericals:
        table_min = np.min(df_orig[numericals], axis=0)
        table_max = np.max(df_orig[numericals], axis=0)


    def distance(x_index, xp_indices):
        def dn(x_register, xp_registers):
            start = timeit.default_timer()

            result = np.abs(x_register - xp_registers) / (table_max - table_min)

            stop = timeit.default_timer()
            # print(f"dn execution time: {stop-start}s")
            return result

        def dc(x_register, xp_registers):
            _dc_cache = dict()
            def compute_dc(row):
                row_values = tuple(row.tolist())
                # print(row_values)
                if row_values in _dc_cache:
                    return _dc_cache[row_values]
                cols = categoricals
                result = np.zeros(len(cols))
                for i, c in enumerate(cols):
                    root = att_tree[c]['*']
                    lca = root.lca(att_tree[c][x_register[c]], att_tree[c][row[c]])
                    # print(lca.value, x_register[c], row[c])
                    parent_leaves = lca.leaf_num
                    tree_leaves = root.leaf_num
                    result[i] = parent_leaves / tree_leaves
                    # print(x_register[c], row[c], lca.value)
                    # print(parent_leaves, tree_leaves)
                # print(result)
                _dc_cache[row_values] = result
                return result

            start = timeit.default_timer()

            # results = np.zeros((xp_registers.shape[0], len(categoricals)))
            # for i in range(xp_registers.shape[0]):
            #     for j, c in enumerate(categoricals):
            #         root = att_tree[c]['*']
            #         parent_leaves = root.lca(att_tree[c][x_register[c]], att_tree[c][xp_registers.loc[i, c]]).leaf_num
            #         tree_leaves = root.leaf_num
            #         results[i][j] = parent_leaves / tree_leaves

            # result = xp_registers.swifter.progress_bar(False).apply(compute_dc, axis=1, result_type='expand')
            result = xp_registers.apply(compute_dc, axis=1, result_type='expand')
            # print(result)
            results = np.where(x_register == xp_registers, 0, result)
            # print(results)
            stop = timeit.default_timer()
            # print(f"dc execution time: {stop-start}s")
            return results

        x_register = df_anon.loc[x_index]
        xp_registers = df_orig.loc[xp_indices]

        if numericals:
            if num_intervals:
                x_numericals = np.zeros(len(numericals))
                for i, num in enumerate(numericals):
                    value = x_register[num]

                    if value == '*':
                        x_numericals[i] = int(np.mean((table_min.iloc[i], table_max.iloc[i])))
                    elif '-' in value:
                        n1, n2 = value.split('-')
                        n1 = int(n1)
                        n2 = int(n2)
                        x_numericals[i] = int(np.mean((n1, n2)))
                    else:
                        x_numericals[i] = int(value)

            else:
                x_numericals = x_register[numericals]

            xp_numericals = xp_registers[numericals]
            dn_value = np.sum(dn(x_numericals, xp_numericals), axis=1)
        else:
            dn_value = 0


        if categoricals:
            dc_value = np.sum(dc(x_register[categoricals], xp_registers[categoricals]), axis=1)
        else:
            dc_value = 0

        result = dn_value + dc_value
        # result = np.sum(dn(x_register[numericals], xp_registers[numericals]), axis=1) + np.sum(dc(x_register[categoricals], xp_registers[categoricals]), axis=1)
        # print(result)
        return result


    x = df_anon
    xp = df_orig

    match = 0


    for x_index in range(x.shape[0]):
        # Given a record x_register in x, we compute its distance to each record r2 in xp
        # dists = [distance(x_index, i_xp) for i_xp in range(xp.shape[0])]
        dists = distance(x_index, xp.index)
        # print(dists)

        # Then, we select the most similar to x_register
        i_xp = np.argmin(dists)

        # print(f'{x_index} of {x.shape[0]}')

        if (x_index == i_xp):
            match += 1


    return (match, len(x))



# def record_linkage(df_orig, df_anon, numericals, categoricals, num_intervals=False):
#     table_min = np.min(df_orig[numericals], axis=0)
#     table_max = np.max(df_orig[numericals], axis=0)

#     def distance(x_index, xp_indices):
#         def dn(x_register, xp_registers):
#             # start = timeit.default_timer()

#             result = np.abs(x_register - xp_registers) / (table_max - table_min)

#             # stop = timeit.default_timer()
#             # print(f"dn execution time: {stop-start}s")
#             return result

#         def dc(x_register, xp_registers):
#             # start = timeit.default_timer()
#             results = np.where(x_register == xp_registers, 0, 1)
#             # stop = timeit.default_timer()
#             # print(f"dc execution time: {stop-start}s")
#             return results

#         x_register = df_anon.loc[x_index]
#         xp_registers = df_orig.loc[xp_indices]

#         if num_intervals:
#             x_numericals = np.zeros(len(numericals))
#             for i, num in enumerate(numericals):
#                 value = x_register[num]

#                 if value == '*':
#                     x_numericals[i] = int(np.mean((table_min.iloc[i], table_max.iloc[i])))
#                 elif '-' in value:
#                     n1, n2 = value.split('-')
#                     n1 = int(n1)
#                     n2 = int(n2)
#                     x_numericals[i] = int(np.mean((n1, n2)))
#                 else:
#                     x_numericals[i] = int(value)

#         else:
#             x_numericals = x_register[numericals]

#         xp_numericals = xp_registers[numericals]

#         result = np.sum(dn(x_numericals, xp_numericals), axis=1) + np.sum(dc(x_register[categoricals], xp_registers[categoricals]), axis=1)
#         # print(result)
#         return result




#     x = df_anon.copy()
#     xp = df_orig.copy()

#     # for cat in categoricals:
#     #     xp.loc[:, cat] = xp[cat].apply(lambda value: att_tree[cat][value].parent[0].value)

#     # encoder = MyLabelEncoder()
#     # encoder.encode(xp)
#     # encoder.encode(x)

#     match = 0

#     for x_index in range(x.shape[0]):
#         # Given a record x_register in x, we compute its distance to each record r2 in xp
#         # dists = [distance(x_index, i_xp) for i_xp in range(xp.shape[0])]
#         dists = distance(x_index, xp.index)

#         # Then, we select the most similar to x_register
#         i_xp = np.argmin(dists)

#         # print(f'{x_index} of {x.shape[0]}')

#         if (x_index == i_xp):
#             match += 1

#     return (match, len(x))
