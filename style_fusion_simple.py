import json
import sys

import torch
import torch.nn.functional

sys.path.append(".")
sys.path.append("..")

from stylefusion.sf_stylegan2 import SFGenerator
from stylefusion.sf_hierarchy import SFHierarchyFFHQ, SFHierarchyCar, SFHierarchyChurch
from PIL import Image


def tensor2im(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))


class StyleFusionSimple:
    def __init__(self, stylegan_type, stylegan_weights, fusion_nets_weights):
        self.stylegan_type = stylegan_type
        if self.stylegan_type == "ffhq":
            self.truncation = 0.7
            self.stylegan_size = 1024
            self.stylegan_layers = 18
        elif self.stylegan_type == "car":
            self.truncation = 0.5
            self.stylegan_size = 512
            self.stylegan_layers = 16
        elif self.stylegan_type == "church":
            self.truncation = 0.5
            self.stylegan_size = 256
            self.stylegan_layers = 14

        self.device = 'cuda:0'

        stylegan_ckpt = torch.load(stylegan_weights, map_location='cpu')
        self.original_net = SFGenerator(self.stylegan_size, 512, 8)
        self.original_net.load_state_dict(stylegan_ckpt['g_ema'], strict=True)

        self.original_net.to(self.device)

        with torch.no_grad():
            self.mean_latent = self.original_net.mean_latent(4096)

        if self.stylegan_type == "ffhq":
            self.sf_hierarchy = SFHierarchyFFHQ()
            self.base_blender = self.sf_hierarchy.nodes["all"]
        elif self.stylegan_type == "car":
            self.sf_hierarchy = SFHierarchyCar()
            self.base_blender = self.sf_hierarchy.nodes["all"]
        elif self.stylegan_type == "church":
            self.sf_hierarchy = SFHierarchyChurch()
            self.base_blender = self.sf_hierarchy.nodes["all"]

        with open(fusion_nets_weights, 'r') as f:
            fusion_nets_paths = json.load(f)

        keys = fusion_nets_paths.keys()

        for key in keys:
            self.sf_hierarchy.nodes[key].load_fusion_net(fusion_nets_paths[key])
            self.sf_hierarchy.nodes[key].fusion_net.to(self.device)
            self.sf_hierarchy.nodes[key].fusion_net.eval()

    def generate_img(self, base_latent, latents_type="z", hair=None, face=None, background=None, all=None, mouth=None,
                            eyes=None, wheels=None, car=None, bg_top=None, bg_bottom=None):
        s_dict = dict()
        parts = self.sf_hierarchy.nodes["all"].get_all_active_parts()
        for part in parts:
            s_dict[part] = self.general_latent_to_s(base_latent, latents_type)

        def swap(value, keys):
            if value is None:
                return
            for k in keys:
                s_dict[k] = self.general_latent_to_s(value, latents_type)

        swap(hair, ["bg_hair_clothes", "hair"])
        swap(face, ["face", "eyes", "skin_mouth", "mouth", "skin", "shirt"])
        swap(background, ["background", "background_top", "background_bottom", "bg"])
        swap(all, ["all"])
        swap(mouth, ["skin_mouth", "face"])
        swap(eyes, ["eyes", "face"])
        swap(wheels, ["wheels"])
        swap(car, ["car", "body", "wheels", "car_body"])
        swap(bg_top, ["background_top"])
        swap(bg_bottom, ["background_bottom"])

        return self.s_dict_to_image(s_dict)[0]

    def seed_to_z(self, seed):
        torch.manual_seed(seed[0])
        z_regular = torch.randn((seed[1] + 1, 1, 512), device=self.device)
        return z_regular[seed[1]]

    def z_to_s(self, z):
        return self.original_net([z],
                                     truncation=self.truncation, truncation_latent=self.mean_latent,
                                     randomize_noise=False, return_style_vector=True)

    def z_to_w_plus(self, z):
        _, res = self.original_net([z],
                                     truncation=self.truncation, truncation_latent=self.mean_latent,
                                     randomize_noise=False, return_latents=True)
        return res[0]

    def w_plus_to_s(self, w_plus, truncation):
        return self.original_net([w_plus], input_is_latent=True,
                                     truncation=truncation, truncation_latent=self.mean_latent,
                                     randomize_noise=False, return_style_vector=True)

    def general_latent_to_s(self, l, latent_type):
        assert latent_type in ["z", "w", "w+", "s"]

        if latent_type == "z":
            return self.z_to_s(l)
        elif latent_type == "w" or latent_type == "w+":
            if l.size()[0] != self.stylegan_layers:
                return self.w_plus_to_s(l.unsqueeze(0).repeat(self.stylegan_layers, 1), truncation=1)
            else:
                return self.w_plus_to_s(l, truncation=1)
        else:
            return l

    def s_to_image(self, s):
        img, _ = self.original_net([torch.zeros(1, 512, device=self.device)],
                                                          randomize_noise=False, style_vector=s)
        return img

    def w_plus_to_image(self, w_plus):
        s = self.w_plus_to_s(w_plus, truncation=1)
        return self.s_to_image(s)

    def z_to_image(self, z):
        s = self.z_to_s(z)
        return self.s_to_image(s)

    def s_dict_to_image(self, s_dict):
        s = self.base_blender.forward(s_dict)
        return self.s_to_image(s)

    def w_plus_dict_to_image(self, w_plus_dict, truncation=1):
        s_dict = dict()
        for key in w_plus_dict.keys():
            s_dict[key] = self.w_plus_to_s(w_plus_dict[key], truncation=truncation)
        return self.s_dict_to_image(s_dict)

    def z_dict_to_image(self, z_dict):
        s_dict = dict()
        for key in z_dict.keys():
            s_dict[key] = self.z_to_s(z_dict[key])
        return self.s_dict_to_image(s_dict)
