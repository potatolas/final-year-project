import libcst as cst

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