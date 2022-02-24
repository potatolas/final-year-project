import libcst as cst
import numpy as np
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt
from itertools import cycle

class MethodCall:
  def __init__(self, name, node):
    self.name = name
    self.node = node

class UsagePattern:
  def __init__(self, method, pre_set, post_set, accompany_set):
    self.method = method
    self.pre_set = pre_set
    self.post_set = post_set
    self.accompany_set = accompany_set

  def print(self):
        print("========== Usage Pattern ==========")
        print("Target Method: " + self.method.name)
        print("Pre-Set: {", end ="")
        for i in range(len(self.pre_set)):
            if(i != len(self.pre_set) - 1):
                print(self.pre_set[i].name, end=", ")
            else:
                print(self.pre_set[i].name, end="}")
                print()
        if(len(self.pre_set) == 0):
            print("}")
        print("Post-Set: {", end ="")
        for i in range(len(self.post_set)):
            if(i != len(self.post_set) - 1):
                print(self.post_set[i].name, end=", ")
            else:
                print(self.post_set[i].name, end="}")
                print()
        if(len(self.post_set) == 0):
            print("}")
        print("Accompany-Set: {", end ="")
        for i in range(len(self.accompany_set)):
            if(i != len(self.accompany_set) - 1):
                print(self.accompany_set[i], end=", ")
            else:
                print(self.accompany_set[i], end="}")
                print()
        if(len(self.accompany_set) == 0):
            print("}")
        print("===================================")

def sort_statements(statements):
    simple_statement_lines = [] #list containing nodes of type SimpleStatementLines
    function_def_lines = [] #list containing nodes of type FunctionDef
    class_def_lines = [] #list containing nodes of type ClassDef

    for statement in statements:
        if(isinstance(statement, cst.SimpleStatementLine)):
            simple_statement_lines.append(statement)
            function_def_lines.append("Simple Statement")
        elif (isinstance(statement, cst.FunctionDef)):
            function_def_lines.append(statement)
            simple_statement_lines.append("Function Def")
        elif (isinstance(statement, cst.ClassDef)):
            class_def_lines.append(statement)
    
    return [simple_statement_lines, function_def_lines, class_def_lines]

def cluster(data, maxgap):
    data.sort()
    groups = [[data[0]]]
    for x in data[1:]:
        if abs(x - groups[-1][-1]) <= maxgap:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups

def search_call_sequence(target_method, call_sequences_processed):
    target_call_sequence = []
    for call_sequence in call_sequences_processed:
        for call in call_sequence:
            if(call.name == target_method.name):
                target_call_sequence.append(call_sequence)
                break
    return target_call_sequence

def find_common_methods(listA, listB):
    list1 = []
    list2 = []

    for method in listA:
        list1.append(method.name)
    for method in listB:
        list2.append(method.name)

    return find_common_of_two_list(list1, list2)

def find_common_of_two_list(list1, list2):
    list1_as_set = set(list1)
    intersection = list1_as_set.intersection(list2)
    intersection_as_list = list(intersection)
    return intersection_as_list

def generate_usage_pattern(target_method, target_sequence):

    call_count = 0
    indexes = []
    for i in range(len(target_sequence)):
        if(target_method.name == target_sequence[i].name):
            call_count = call_count + 1
            indexes.append(i)
    
    pre_set = []
    post_set = []
    accompany_set = []

    pre_set_name = []
    post_set_name = []
    accompany_set_name = []
    
    if(call_count == 1):
        for i in range(0, indexes[0]):
            if(target_sequence[i].name not in pre_set_name):
                pre_set.append(target_sequence[i])
                pre_set_name.append(target_sequence[i].name)
        for i in range(indexes[0]+1, len(target_sequence)):
            if(target_sequence[i].name not in post_set_name):
                post_set.append(target_sequence[i])
                post_set_name.append(target_sequence[i].name)
    else:
        for i in range(0, max(indexes)):
            if(target_sequence[i].name != target_method.name and target_sequence[i].name not in pre_set_name):
                pre_set.append(target_sequence[i])
                pre_set_name.append(target_sequence[i].name)
        for i in range(min(indexes)+1, len(target_sequence)):
            if(target_sequence[i].name != target_method.name and target_sequence[i].name not in post_set_name):
                post_set.append(target_sequence[i])
                post_set_name.append(target_sequence[i].name)

        before_list = []
        after_list = []

        for i in range(len(indexes)):
            if(i == 0):
                before_list.append(target_sequence[0:indexes[i]])
            else:
                before_list.append(target_sequence[indexes[i-1]+1:indexes[i]])

        for listA in before_list:
            for listB in before_list:
                temp_list = []
                if(listA != listB):
                    temp_list = find_common_methods(listA, listB)
                for method_name in temp_list:
                    if(method_name not in accompany_set_name and method_name != target_method.name):
                        accompany_set_name.append(method_name)

        after_list = before_list[1:]
        after_list.append(target_sequence[indexes[len(indexes) - 1] + 1:])
        
        for listA in after_list:
            for listB in after_list:
                temp_list = []
                if(listA != listB):
                    temp_list = find_common_methods(listA, listB)
                for method_name in temp_list:
                    if(method_name not in accompany_set_name and method_name != target_method.name):
                        accompany_set_name.append(method_name)

    
    return UsagePattern(target_method, pre_set, post_set, accompany_set_name)

def method_to_string_list(list_of_method):
    result = []
    for method in list_of_method:
        result.append(method.name)
    return result
    
def jaccard_similarity(list1, list2):

    if(len(list1) == 0 and len(list2) == 0):
        return 1 

    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return float(intersection) / union

def constructing_2d_matrix(size):
    similarity_matrix = []
    for row in range(0, size):
        similarity_matrix.append([])
        for col in range(0, size):
            similarity_matrix[row].append(-1)

    return similarity_matrix

def get_matrix_value(usage_pattern_1, usage_pattern_2):
    row_preset_names = method_to_string_list(usage_pattern_1.pre_set)
    col_preset_names = method_to_string_list(usage_pattern_2.pre_set)

    row_postset_names = method_to_string_list(usage_pattern_1.post_set)
    col_postset_names = method_to_string_list(usage_pattern_2.post_set)

    pre_value = jaccard_similarity(row_preset_names, col_preset_names)
    post_value = jaccard_similarity(row_postset_names, col_postset_names)
    accompany_value = jaccard_similarity(usage_pattern_1.accompany_set, usage_pattern_2.accompany_set)

    matrix_value = (pre_value + post_value + accompany_value)/3
    return matrix_value

def generate_similarity_matrix(target_usage_pattern_list):
    similarity_matrix = constructing_2d_matrix(len(target_usage_pattern_list))
    for row in range(0, len(target_usage_pattern_list)):
        for col in range(0, len(target_usage_pattern_list)):
            if(row == col):
                similarity_matrix[row][col] = 1
            if(similarity_matrix[row][col] == -1):
                matrix_value = get_matrix_value(target_usage_pattern_list[row], target_usage_pattern_list[col])
                similarity_matrix[row][col] = matrix_value
                similarity_matrix[col][row] = matrix_value
    
    return similarity_matrix

def generate_clusters(similarity_matrix, plot_flag):

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning) 

    X = np.array(similarity_matrix)
    af = AffinityPropagation(affinity="precomputed", damping=0.5, convergence_iter=7, max_iter=400).fit(X)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_

    n_clusters_ = len(cluster_centers_indices)

    if(plot_flag):
        plt.close("all")
        plt.figure(1)
        plt.clf()

        colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
        for k, col in zip(range(n_clusters_), colors):
            class_members = labels == k
            cluster_center = X[cluster_centers_indices[k]]
            plt.plot(X[class_members, 0], X[class_members, 1], col + ".")
            plt.plot(
                cluster_center[0],
                cluster_center[1],
                "o",
                markerfacecolor=col,
                markeredgecolor="k",
                markersize=14,
            )
            for x in X[class_members]:
                plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

        
        plt.title("Estimated number of clusters: %d" % n_clusters_)
        plt.show()

    print("Labels:")
    print(labels)