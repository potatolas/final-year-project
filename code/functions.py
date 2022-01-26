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

from classes import UsagePattern
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
