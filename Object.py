def decay_func(x):
    return .95 ** x

class Object:
    def __init__(self, obj_type, reward, index, visible):
        self.type = obj_type
        self.reward = reward
        self.visible = visible
        self.to_appear = True
        self.index = index
        self.visible_dur = 0 #.95
        self.invisible_dur = 0
        self.invisible_steps_range = [1, 10]
