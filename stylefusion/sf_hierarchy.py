import torch

from stylefusion.fusion_net import FusionNet


class SFHierarchyFFHQ:
    def __init__(self):
        self.nodes = dict()
        self.nodes["clothes"] = SFNode("clothes")
        self.nodes["mouth"] = SFNode("mouth")
        self.nodes["eyes"] = SFNode("eyes")
        self.nodes["bg"] = SFNode("bg")
        self.nodes["hair"] = SFNode("hair")
        self.nodes["skin"] = SFNode("skin")
        self.nodes["skin_mouth"] = SFNode("skin_mouth", child1=self.nodes["mouth"], child2=self.nodes["skin"])
        self.nodes["face"] = SFNode("face", child1=self.nodes["skin_mouth"], child2=self.nodes["eyes"])
        self.nodes["bg_clothes"] = SFNode("bg_clothes", child1=self.nodes["clothes"], child2=self.nodes["bg"])
        self.nodes["bg_hair_clothes"] = SFNode("bg_hair_clothes",
                                               child1=self.nodes["bg_clothes"], child2=self.nodes["hair"])
        self.nodes["all"] = SFNode("all", child1=self.nodes["face"], child2=self.nodes["bg_hair_clothes"])


class SFHierarchyCar:
    def __init__(self):
        self.nodes = dict()
        self.nodes["wheels"] = SFNode("wheels")
        self.nodes["car_body"] = SFNode("car_body")
        self.nodes["background_top"] = SFNode("background_top")
        self.nodes["background_bottom"] = SFNode("background_bottom")
        self.nodes["car"] = SFNode("car", child1=self.nodes["car_body"], child2=self.nodes["wheels"])
        self.nodes["background"] = SFNode("background",
                                          child1=self.nodes["background_top"], child2=self.nodes["background_bottom"])
        self.nodes["all"] = SFNode("all", child1=self.nodes["car"], child2=self.nodes["background"])


class SFHierarchyChurch:
    def __init__(self=None):
        self.nodes = dict()
        self.nodes["church"] = SFNode("church")
        self.nodes["background"] = SFNode("background")
        self.nodes["all"] = SFNode("all", child1=self.nodes["church"], child2=self.nodes["background"])


class SFNode:
    def __init__(self, name, child1=None, child2=None):
        self.name = name
        self.child1 = child1
        self.child2 = child2
        self.fusion_net = None
        if child1 is None or child2 is None:
            assert child1 is None and child2 is None
            self._leaf = True
        else:
            self._leaf = False

    def get_all_parts(self):
        if self._leaf:
            return [self.name]
        else:
            return [self.name] + self.child1.get_all_parts() + self.child2.get_all_parts()

    def get_all_active_parts(self):
        if self.fusion_net is None:
            return [self.name]
        else:
            return [self.name] + self.child1.get_all_active_parts() + self.child2.get_all_active_parts()

    def get_fusion_nets_amount(self):
        if self.fusion_net is None:
            return 0
        if self._leaf:
            return 1
        else:
            return 1 + self.child1.get_fusion_nets_amount() + self.child2.get_fusion_nets_amount()

    def get_fusion_nets(self):
        if self.fusion_net is None:
            return []
        if self._leaf:
            return [self.fusion_net]
        else:
            return [self.fusion_net] + self.child1.get_fusion_nets() + self.child2.get_fusion_nets()

    def forward(self, s_dict):
        if self.fusion_net is None:
            return s_dict[self.name]

        if not (self.child1.name in s_dict.keys() and self.child2.name in s_dict.keys()):
            return s_dict[self.name]

        return self.fusion_net(
            self.child1.forward(s_dict),
            self.child2.forward(s_dict),
            s_dict[self.name])

    def load_fusion_net(self, path):
        data = torch.load(path)

        self.fusion_net = FusionNet()

        if "state_dict" in data.keys():
            data = data["state_dict"]

        self.fusion_net.load_state_dict(data)
