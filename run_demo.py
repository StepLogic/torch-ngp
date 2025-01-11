import copy
import torch
import argparse
from nerf.provider import NeRFDataset
from nerf.foc_gui import NeRFGUI
from nerf.utils import *


import torch
import tqdm
import numpy as np
import os
from PIL import Image
from nerf.utils import Trainer
import raymarching

class MultiNeRFTrainer:
    def __init__(self, trainers, device="cuda"):
        """
        Initialize with multiple trainer instances
        trainers: list of Trainer instances, each managing a different NeRF model
        """
        self.trainers = trainers
        self.device = device
        self.models = list(self.trainers.keys())
        self.scene_composition=[]
        self.scene_editing={
            "fixed_pose":None,
            "static_objects":[]
        }
        self.active_trainer = list(trainers.values())[0]  # Use first trainer's properties for compatibility
        # Copy some properties from active trainer for compatibility
        # self.model = self.active_trainer.model
        # self.workspace = self.active_trainer.workspace
        self.opt = self.active_trainer.opt
        # self.opt.num_steps=512
        # print(trainers)

        

    # def best_densities_and_colors(self, densities, max_densities, rgbs, best_rgbs):
    #     """Combine densities and colors based on highest density values"""
    #     new_max_densities = torch.maximum(densities, max_densities)
    #     new_best_rgbs = torch.where(densities[..., None] > max_densities[..., None], rgbs, best_rgbs)
    #     return new_max_densities, new_best_rgbs

    # def get_combined_output(self, rays_o, rays_d, staged=True, **kwargs):
    #     """Get combined output from all trainers"""
    #     max_densities = None
    #     max_rgbs = None
    #
    #     for trainer in self.trainers:
    #         with torch.cuda.amp.autocast(enabled=trainer.fp16):
    #             results = trainer.model.get_raw_values(rays_o, rays_d, staged=staged, **kwargs)
    #             # print(results.keys())
    #             max_densities = results['densities']
    #             max_rgbs = results['rgbs']
    #             # if max_densities is None:
    #             #     max_densities = densities
    #             #     max_rgbs = rgbs
    #             # else:
    #             #     max_densities, max_rgbs = self.best_densities_and_colors(
    #             #         densities, max_densities, rgbs, max_rgbs
    #             #     )
    #
    #     return {
    #         'sigma': max_densities,
    #         'rgbs': max_rgbs,
    #     }
    # def image_depth_generation(self, rays_o, rays_d,sigmas,rgbs,dt_gamma=0, bg_color=None, perturb=False, force_all_rays=False, max_steps=1024, T_thresh=1e-4, **kwargs):
    #     # rays_o, rays_d: [B, N, 3], assumes B == 1
    #     # return: image: [B, N, 3], depth: [B, N]

    #     prefix = rays_o.shape[:-1]
    #     rays_o = rays_o.contiguous().view(-1, 3)
    #     rays_d = rays_d.contiguous().view(-1, 3)

    #     N = rays_o.shape[0] # N = B * N, in fact
    #     device = rays_o.device
        

    #     # pre-calculate near far
    #     nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d,self.active_trainer.model.aabb_infer, self.active_trainer.model.min_near)

    #     # mix background color
    #     if self.active_trainer.model.bg_radius > 0:
    #         # use the bg model to calculate bg_color
    #         sph = raymarching.sph_from_ray(rays_o, rays_d, self.active_trainer.model.bg_radius) # [N, 2] in [-1, 1]
    #         bg_color = self.active_trainer.model.background(sph, rays_d) # [N, 3]
    #     elif bg_color is None:
    #         bg_color = 1

    #     results = {}

        
    #     # allocate outputs 
    #     # if use autocast, must init as half so it won't be autocasted and lose reference.
    #     #dtype = torch.half if torch.is_autocast_enabled() else torch.float32
    #     # output should always be float32! only network inference uses half.
    #     dtype = torch.float32
        
    #     weights_sum = torch.zeros(N, dtype=dtype, device=device)
    #     depth = torch.zeros(N, dtype=dtype, device=device)
    #     image = torch.zeros(N, 3, dtype=dtype, device=device)
        
    #     n_alive = N
    #     rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
    #     rays_t = nears.clone() # [N]
    #     # step = 0
    #     # while step < max_steps:

    #         # count alive rays 
    #     # n_alive = rays_alive.shape[0]
        
    #     # # exit loop
    #     # if n_alive <= 0:
    #     #     break

    #     # decide compact_steps
    #     n_step = max(min(N // n_alive, 8), 1)
    #     # breakpoint()
    #     _, _, deltas = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.active_trainer.model.bound, self.active_trainer.model.density_bitfield, self.active_trainer.model.cascade, self.active_trainer.model.grid_size, nears, fars, 128, False , dt_gamma, max_steps)
    #     # sigmas, rgbs = self(xyzs, dirs)
    #     # density_outputs = self.density(xyzs) # [M,], use a dict since it may include extra things, like geo_feat for rgb.
    #     # sigmas = density_outputs['sigma']
    #     # rgbs = self.color(xyzs, dirs, **density_outputs)
    #     # pure_rgbs= rgbs.view(N, -1, 3) # [N, T+t, 3]
    #     # densities=sigmas
    #     sigmas = self.active_trainer.model.density_scale * sigmas

    #     raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image, T_thresh)
    #     # rays_alive = rays_alive[rays_alive >= 0]
    #     #print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}, xyzs: {xyzs.shape}')
    #     # step += n_step

    #     image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
    #     depth = torch.clamp(depth - nears, min=0) / (fars - nears)
    #     image = image.view(*prefix, 3)
    #     depth = depth.view(*prefix)
    #     results["image"]=image
    #     results["depth"]=depth
    #     return results
    # def image_depth_generation(self, rays_o,rays_d, sigmas, rgbs, bg_color=None):
    #     """
    #     Generate image and depth from density and color values
    #     Args:
    #         data: dict containing rays_o and rays_d
    #         sigmas: tensor of shape [N, T] containing density values
    #         rgbs: tensor of shape [N, T, 3] containing color values
    #     """
    #     # Prepare rays
    #     # rays_o = data['rays_o'].contiguous().view(-1, 3)
    #     # rays_d = data['rays_d'].contiguous().view(-1, 3)
    #     prefix = rays_o.shape[:-1]
    #     rays_o = rays_o.contiguous().view(-1, 3)
    #     rays_d = rays_d.contiguous().view(-1, 3)

    #     prefix = rays_o.shape[:-1]
    #     N = rays_o.shape[0]
    #     device = rays_o.device

    #     # Get near-far bounds
    #     aabb = self.active_trainer.model.aabb_infer
    #     nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.active_trainer.model.min_near)
    #     nears = nears.unsqueeze(-1)
    #     fars = fars.unsqueeze(-1)

    #     # Sample points along rays
    #     z_vals = torch.linspace(0.0, 1.0, self.opt.num_steps, device=device).unsqueeze(0)
    #     z_vals = z_vals.expand((N, self.opt.num_steps))
    #     z_vals = nears + (fars - nears) * z_vals
    #     sample_dist = (fars - nears) / self.opt.num_steps

    #     # Calculate weights
    #     deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T-1]
    #     deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
    #     # breakpoint()
    #     # print(deltas.shape,sigmas.shape)
    #     alphas = 1 - torch.exp(-deltas * sigmas)
    #     alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-10], dim=-1)
    #     weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]

    #     # Compute weighted color
    #     weights_sum = weights.sum(dim=-1)
    #     depth = torch.sum(weights * z_vals, dim=-1)
    #     image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2)

    #     # Handle background
    #     if bg_color is None:
    #         bg_color = torch.ones_like(image)
    #     elif isinstance(bg_color, str):
    #         if bg_color == 'white':
    #             bg_color = torch.ones_like(image)
    #         elif bg_color == 'black':
    #             bg_color = torch.zeros_like(image)

    #     image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
    #     depth = torch.clamp(depth - nears.squeeze(-1), min=0) / (fars.squeeze(-1) - nears.squeeze(-1))

    #     # Reshape to original dimensions
    #     image = image.reshape(*prefix, 3).clamp(0, 1)
    #     depth = depth.reshape(*prefix)

    #     return image, depth

    # def image_depth_generation(self, rays_o,rays_d, sigmas, rgbs,dt_gamma=0,max_steps=1024, bg_color=None,T_thresh=1e-4):
    #         dtype = torch.float32
    #         device = rays_o.device
    #         prefix = rays_o.shape[:-1]
    #         rays_o = rays_o.contiguous().view(-1, 3)
    #         rays_d = rays_d.contiguous().view(-1, 3)
    #         N = rays_o.shape[0] # N = B * N, in fact
    #         weights_sum = torch.zeros(N, dtype=dtype, device=device)
    #         depth = torch.zeros(N, dtype=dtype, device=device)
    #         image = torch.zeros(N, 3, dtype=dtype, device=device) 
    #         nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.active_trainer.model.aabb_infer, self.active_trainer.model.min_near)
    #         n_alive = N
    #         rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
    #         rays_t = nears.clone() # [N]
    #         step = 0
    #         perturb=False
    #         while step < max_steps:
    #             n_alive = rays_alive.shape[0]
    #             if n_alive <= 0:
    #                 break
    #             # decide compact_steps
    #             n_step = max(min(N // n_alive, 8), 1)
    #             xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.active_trainer.model.bound, self.active_trainer.model.density_bitfield, self.active_trainer.model.cascade, self.active_trainer.model.grid_size, nears, fars, 128, perturb if step == 0 else False, dt_gamma, max_steps)
    #             del xyzs,dirs
    #             sigmas = self.active_trainer.model.density_scale * sigmas
    #             raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image, T_thresh)
    #             rays_alive = rays_alive[rays_alive >= 0]
    #             step += n_step
    #         del sigmas,rgbs,rays_alive
    #         bg_color = 1
    #         image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
    #         depth = torch.clamp(depth - nears, min=0) / (fars - nears)
    #         image = image.view(*prefix, 3)
    #         depth = depth.view(*prefix)
    #         return image,depth
    

    def blend_nerf_outputs_fast(self, xyzs, dirs, trainers, scene_editing, use_fixed_rays=False):
        """
        Vectorized version of NeRF output blending using painter's algorithm.
        """
        device = xyzs.device
        N = xyzs.shape[0]
        max_trainers = len(trainers)
        # Pre-allocate tensors for all outputs
        all_sigmas = torch.zeros((max_trainers, N), device=device)
        all_rgbs = torch.zeros((max_trainers, N, 3), device=device)
        valid_mask = torch.zeros((max_trainers, N), dtype=torch.bool, device=device)
        # Collect outputs in parallel
        idx = 0
        for key, trainer in trainers.items():
            if key in self.scene_composition:
                continue
            # should_process = use_fixed_rays
            # if not should_process:
            #     should_process = not key in scene_editing["static_objects"]
            # else:
            #     should_process = not key in scene_editing["static_objects"]
            # # else:

            # should_process = not use_fixed_rays or (use_fixed_rays and not key in scene_editing["static_objects"])
            # print(should_process,use_fixed_rays,key in scene_editing["static_objects"],key,scene_editing["static_objects"])
            # if should_process:
            print(key,trainer.opt)
            sigmas, rgbs = trainer.model(xyzs, dirs)
            all_sigmas[idx] = sigmas
            all_rgbs[idx] = rgbs
            valid_mask[idx] = True
            idx += 1
        
        if idx == 0:
            return None, None
        # breakpoint()
            
        # Trim tensors to actual size
        all_sigmas = all_sigmas[:idx]
        all_rgbs = all_rgbs[:idx]
        valid_mask = valid_mask[:idx]
        
        # Calculate alphas
        alphas = 1 - torch.exp(-all_sigmas)  # [num_trainers, N]
        
        # Sort by density for each point
        # densities_sorted, sort_indices = torch.sort(all_sigmas, dim=0, descending=True)  # [num_trainers, N]
        
        # Gather sorted alphas and RGBs
        # alphas_sorted = torch.gather(alphas, 0, sort_indices)  # [num_trainers, N]
        
        # Create indices for gathering RGB values
        # gather_indices = sort_indices.unsqueeze(-1).expand(-1, -1, 3)  # [num_trainers, N, 3]
        # rgbs_sorted = torch.gather(all_rgbs, 0, gather_indices)  # [num_trainers, N, 3]
        
        # Calculate visibility (transparency) for each layer
        visibility = torch.ones_like(all_sigmas)  # [num_trainers, N]
        accumulated_alpha = torch.zeros(N, device=device)  # [N]
        
        final_rgb = torch.zeros((N, 3), device=device)
        final_sigma = torch.zeros(N, device=device)
        
        # Efficient vectorized blending
        for i in range(idx):
            curr_alpha = alphas[i]  # [N]
            curr_rgb = all_rgbs[i]  # [N, 3]
            curr_sigma = all_sigmas[i]  # [N]
            
            # Update visibility
            visibility = 1 - accumulated_alpha
            
            # Update colors and density
            contribution = visibility.unsqueeze(-1) * curr_alpha.unsqueeze(-1) * curr_rgb
            final_rgb += contribution
            final_sigma = torch.maximum(final_sigma, curr_sigma)
            
            # Update accumulated alpha
            accumulated_alpha += visibility * curr_alpha
        
        return final_sigma, final_rgb

    def render_with_blending_fast(self, rays_o, rays_d, fixed_rays_o=None, fixed_rays_d=None, **kwargs):
        """
        Optimized version of the main rendering function.
        """
        dtype = torch.float32
        device = rays_o.device
        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        
        if fixed_rays_o is not None and fixed_rays_d is not None:
            fixed_rays_o = fixed_rays_o.contiguous().view(-1, 3)
            fixed_rays_d = fixed_rays_d.contiguous().view(-1, 3)
        
        N = rays_o.shape[0]
        weights_sum = torch.zeros(N, dtype=dtype, device=device)
        depth = torch.zeros(N, dtype=dtype, device=device)
        image = torch.zeros(N, 3, dtype=dtype, device=device)
        
        nears, fars = raymarching.near_far_from_aabb(
            rays_o, rays_d,
            self.active_trainer.model.aabb_infer,
            self.active_trainer.model.min_near
        )
        
        # Ray marching state
        rays_alive = torch.arange(N, dtype=torch.int32, device=device)
        rays_t = nears.clone()
        
        for step in range(0, kwargs.get('max_steps', 1024), 8):
            n_alive = rays_alive.shape[0]
            if n_alive <= 0:
                break
                
            # March and blend static objects
            xyzs, dirs, deltas = raymarching.march_rays(
                n_alive, min(8, N), rays_alive, rays_t,
                rays_o, rays_d,
                self.active_trainer.model.bound,
                self.active_trainer.model.density_bitfield,
                self.active_trainer.model.cascade,
                self.active_trainer.model.grid_size,
                nears, fars,
                128, kwargs.get('perturb', False) and step == 0,
                kwargs.get('dt_gamma', 0), kwargs.get('max_steps', 1024)
            )
            
            # Process static objects
            sigmas, rgbs = self.blend_nerf_outputs_fast(
                xyzs, dirs, self.trainers,
                self.scene_editing,
                use_fixed_rays=False
            )
            # breakpoint()
            if sigmas is not None:
                raymarching.composite_rays(
                    n_alive, min(8, N), rays_alive, rays_t,
                    sigmas, rgbs, deltas,
                    weights_sum, depth, image,
                    kwargs.get('T_thresh', 1e-4)
                )
            
            # Process dynamic objects with fixed rays
            if fixed_rays_o is not None and fixed_rays_d is not None and len(self.scene_editing["static_objects"]) > 0:
                mod_xyzs, mod_dirs, _ = raymarching.march_rays(
                    n_alive, min(8, N), rays_alive, rays_t,
                    fixed_rays_o, fixed_rays_d,
                    self.active_trainer.model.bound,
                    self.active_trainer.model.density_bitfield,
                    self.active_trainer.model.cascade,
                    self.active_trainer.model.grid_size,
                    nears, fars,
                    128, kwargs.get('perturb', False) and step == 0,
                    kwargs.get('dt_gamma', 0), kwargs.get('max_steps', 1024)
                )
                
                sigmas, rgbs = self.blend_nerf_outputs_fast(
                    mod_xyzs, mod_dirs, self.trainers,
                    self.scene_editing,
                    use_fixed_rays=True
                )
                
                if sigmas is not None:
                    # if deltas is None:
                    #     deltas=mod_deltas
                    raymarching.composite_rays(
                        n_alive, min(8, N), rays_alive, rays_t,
                        sigmas, rgbs, deltas,
                        weights_sum, depth, image,
                        kwargs.get('T_thresh', 1e-4)
                    )
            
            rays_alive = rays_alive[rays_alive >= 0]
        
        # Finalize image
        image = image + (1 - weights_sum).unsqueeze(-1)
        depth = torch.clamp(depth - nears, min=0) / (fars - nears)
        
        return image.view(*prefix, 3), depth.view(*prefix)
    def image_depth_generation(self, rays_o, rays_d,fixed_rays_o=None,fixed_rays_d=None, perturb=False,dt_gamma=0, max_steps=1024, bg_color=None, T_thresh=1e-4):
            dtype = torch.float32
            device = rays_o.device
            prefix = rays_o.shape[:-1]
            rays_o = rays_o.contiguous().view(-1, 3)
            rays_d = rays_d.contiguous().view(-1, 3)
            if not (fixed_rays_d is None and fixed_rays_o is None):
                    fixed_rays_o = fixed_rays_o.contiguous().view(-1, 3)
                    fixed_rays_d = fixed_rays_d.contiguous().view(-1, 3)
            
            N = rays_o.shape[0]

            # Initialize outputs
            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)

            # Get near-far bounds
            nears, fars = raymarching.near_far_from_aabb(
                rays_o, rays_d, 
                self.active_trainer.model.aabb_infer,
                self.active_trainer.model.min_near
            )

            # Initialize ray marching
            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device)
            rays_t = nears.clone()
            step = 0

            while step < max_steps:
                n_alive = rays_alive.shape[0]
                if n_alive <= 0:
                    break
                # Decide compact steps
                n_step = max(min(N // n_alive, 8), 1)
                # March rays
                if not (fixed_rays_d is None and fixed_rays_o is None):
                    xyzs, dirs, deltas = raymarching.march_rays(
                        n_alive, n_step,
                        rays_alive, rays_t,
                        fixed_rays_o, fixed_rays_d,
                        self.active_trainer.model.bound,
                        self.active_trainer.model.density_bitfield,
                        self.active_trainer.model.cascade,
                        self.active_trainer.model.grid_size,
                        nears, fars,
                        128,  perturb if step == 0 else False,
                        dt_gamma, max_steps
                    )
                else:
                    xyzs, dirs, deltas = raymarching.march_rays(
                        n_alive, n_step,
                        rays_alive, rays_t,
                        rays_o, rays_d,
                        self.active_trainer.model.bound,
                        self.active_trainer.model.density_bitfield,
                        self.active_trainer.model.cascade,
                        self.active_trainer.model.grid_size,
                        nears, fars,
                        128,  perturb if step == 0 else False,
                        dt_gamma, max_steps
                    )
                # Get outputs from all trainers and combine
                max_sigmas = None
                max_rgbs = None
                # print("is fixed pose avaiblie",not (fixed_rays_d is None and fixed_rays_o is None))
                for key,trainer in self.trainers.items():
                    if key in self.scene_composition:
                        continue
                    # if not key in self.scene_editing["static_objects"]  and not self.scene_editing["fixed_pose"] is None:
                    #     # print("skipping object",max_sigmas,key in self.scene_editing["static_objects"])
                    #     continue
                    sigmas, rgbs = trainer.model(xyzs, dirs)
                    if max_sigmas is None:
                        max_sigmas = sigmas
                        max_rgbs = rgbs
                    else:
                        mask=sigmas > max_sigmas
                        # print("compositing densities",mask)
                        max_sigmas = torch.where(mask, sigmas, max_sigmas)
                        max_rgbs = torch.where(mask.unsqueeze(-1),rgbs , max_rgbs)
   
                # if len(self.scene_editing["static_objects"])>0:
                #     xyzs, dirs, deltas = raymarching.march_rays(
                #         n_alive, n_step,
                #         rays_alive, rays_t,
                #         rays_o, rays_d,
                #         self.active_trainer.model.bound,
                #         self.active_trainer.model.density_bitfield,
                #         self.active_trainer.model.cascade,
                #         self.active_trainer.model.grid_size,
                #         nears, fars,
                #         128,  perturb if step == 0 else False,
                #         dt_gamma, max_steps
                #     )
                #     for key,trainer in self.trainers.items():
                #         if not key in self.scene_editing["static_objects"]:
                #             # print("editing object",key)
                #             sigmas, rgbs = trainer.model(xyzs, dirs)
                #             if max_sigmas is None:
                #                 max_sigmas = sigmas
                #                 max_rgbs = rgbs
                #             else:
                #                 mask=sigmas > max_sigmas
                #                 # breakpoint()
                #                 print("compositing dynamic densities",mask)
                #                 max_sigmas = torch.where(mask, sigmas, max_sigmas)
                #                 max_rgbs = torch.where(mask.unsqueeze(-1),rgbs , max_rgbs)
                            # pass
                # Composite rays using maximum values
                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, max_sigmas, max_rgbs, deltas, weights_sum, depth, image, T_thresh)
                # Update alive rays
                rays_alive = rays_alive[rays_alive >= 0]
                step += n_step
                # Clean up
                del xyzs, dirs, max_sigmas, max_rgbs

            # Add background
            bg_color = 1
            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            depth = torch.clamp(depth - nears, min=0) / (fars - nears)
            # Reshape outputs
            image = image.view(*prefix, 3)
            depth = depth.view(*prefix)
            return image, depth
    # def image_depth_generation(self, rays_o,rays_d, sigmas, rgbs,dt_gamma=0, bg_color=None):
    #     """
    #     Generate image and depth using CUDA raymarching
    #     Args:
    #         data: dict with rays_o and rays_d
    #         densities: tensor of shape [B, N, steps]
    #         rgbs: tensor of shape [B, N, steps, 3]
    #     """
    #     # rays_o = data['rays_o'].contiguous().view(-1, 3)  # [N, 3]
    #     # rays_d = data['rays_d'].contiguous().view(-1, 3)  # [N, 3]
    #     prefix = rays_o.shape[:-1]
    #     rays_o = rays_o.contiguous().view(-1, 3)
    #     rays_d = rays_d.contiguous().view(-1, 3)

    #     # prefix = rays_o.shape[:-1]  # [B, N]
    #     N = rays_o.shape[0]
    #     device = rays_o.device

    #     # Get near-far bounds
    #     nears, fars = raymarching.near_far_from_aabb(
    #         rays_o, rays_d,
    #         self.active_trainer.model.aabb_infer,
    #         self.active_trainer.model.min_near
    #     )

    #     # Setup for ray marching
    #     dtype = torch.float32
    #     weights_sum = torch.zeros(N, dtype=dtype, device=device)
    #     depth = torch.zeros(N, dtype=dtype, device=device)
    #     image = torch.zeros(N, 3, dtype=dtype, device=device)

    #     n_alive = N
    #     rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device)
    #     rays_t = nears.clone()

    #     # Step counter for density grid update
    #     step = 0
    #     max_steps = self.opt.max_steps

    #     # Ray march with CUDA implementation
    #     while step < max_steps:
    #         n_alive = rays_alive.shape[0]
    #         if n_alive <= 0:
    #             break
    #         # Decide number of steps to march
    #         n_step = max(min(N // n_alive, 8), 1)
    #         # March rays using CUDA implementation
    #         xyzs, dirs, deltas = raymarching.march_rays(
    #             n_alive, n_step,
    #             rays_alive, rays_t,
    #             rays_o, rays_d,
    #             self.active_trainer.model.bound,
    #             self.active_trainer.model.density_bitfield,
    #             self.active_trainer.model.cascade,
    #             self.active_trainer.model.grid_size,
    #             nears, fars,
    #             128, False,
    #             dt_gamma,
    #             max_steps
    #         )
    #         # Get density and rgb values for current points
    #         sigmas = sigmas.view(-1, self.opt.num_steps)
    #         rgbs_view = rgbs.view(-1, self.opt.num_steps, 3)

    #         # March step indices
    #         start_idx = step * n_step
    #         end_idx = min(start_idx + n_step, self.opt.num_steps)
    #         current_sigmas = sigmas[:, start_idx:end_idx].contiguous()
    #         current_rgbs = rgbs_view[:, start_idx:end_idx].contiguous()

    #         # Composite rays using CUDA implementation
    #         raymarching.composite_rays(
    #             n_alive, n_step,
    #             rays_alive, rays_t,
    #             current_sigmas, current_rgbs, deltas,
    #             weights_sum, depth, image,
    #             1e-4  # T_threshold
    #         )

    #         # Update alive rays
    #         rays_alive = rays_alive[rays_alive >= 0]

    #         step += n_step

    #     # Handle background
    #     if bg_color == 'white':
    #         bg_color = torch.ones_like(image)
    #     else:  # black
    #         bg_color = torch.zeros_like(image)

    #     # Composite background
    #     image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

    #     # Normalize depth
    #     depth = torch.clamp(depth - nears, min=0) / (fars - nears)

    #     # Reshape outputs
    #     image = image.view(*prefix, 3)
    #     depth = depth.view(*prefix)

    #     return image, depth
    # def image_depth_generation(self, data, densities, rgbs,bg_color="white"):
    #         prefix = data['rays_o'].shape[:-1]
    #         rays_o = data['rays_o'].contiguous().view(-1, 3)  # masked [num_rays,3]  [177,3]
    #         rays_d = data['rays_d'].contiguous().view(-1, 3)
    #         N = data['rays_o'].shape[0]
    #         device = rays_o.device
    #         aabb = self.model.aabb_train if self.model.training else self.model.aabb_infer
    #         nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.model.min_near)
    #         nears.unsqueeze_(-1)
    #         fars.unsqueeze_(-1)
    #         z_vals = torch.linspace(0.0, 1.0, self.opt.num_steps, device=device).unsqueeze(0) # [1, T]
    #         z_vals = z_vals.expand((N,self.opt.num_steps)) # [N, T]
    #         z_vals = nears + (fars - nears) * z_vals
    #         sample_dist = (fars - nears) / self.opt.num_steps # [177,1]
    #         perturb=False
    #         density_scale=1
    #         # bg_color=1
    #         if perturb:
    #             z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
    #         deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T+t-1]
    #         deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
    #         alphas = 1 - torch.exp(-deltas * density_scale * densities.squeeze(-1)) # [N, T+t]
    #         alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+t+1]
    #         weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T+t]
            
    #         rgbs= rgbs.squeeze(0)
    #         weights = weights.squeeze(0)
            
    #         weights_sum = weights.sum(dim=-1)
    #         densities= densities.squeeze(0). unsqueeze(-1)
            
    #         ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)   # [num_rays, num_bins] 177,512]
    #         depth = torch.sum(weights * ori_z_vals, dim=-1)     # 177
    #         image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2)
    #         alpha_channel = torch.sum(weights.unsqueeze(-1) * densities, dim=-2)
           
    #         image = torch.cat((image, alpha_channel), dim=-1) 
        
    #         if bg_color == 'white':
    #             bg_color = torch.ones(N, 1).to(device)
    #         elif bg_color == 'black':
    #             bg_color = torch.zeros(N, 1).to(device)
    #         image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
    #         image = image.clamp(0, 1)
    #         pred_rgb = image
    #         pred_depth = depth
            
    #         return pred_rgb, pred_depth
    # def test_step(self, data, bg_color=None, perturb=False):
    #     """Test step combining outputs from all trainers"""
    #     rays_o = data['rays_o']  # [B, N, 3]
    #     rays_d = data['rays_d']  # [B, N, 3]
    #     H, W = data['H'], data['W']

    #     if bg_color is not None:
    #         bg_color = bg_color.to(self.device)

    #     # Get combined output
    #     kwargs = {
    #         'bg_color': bg_color,
    #         'perturb': perturb,
    #         **vars(self.opt)
    #     }
    #     outputs = self.get_combined_output(rays_o, rays_d, **kwargs)
        
    #     # Process output
    #     # pred_rgb = outputs['rgb'].reshape(-1, H, W, 3)
    #     # pred_depth = outputs['sigma'].mean(dim=-1).reshape(-1, H, W)
    #     pred_rgb, pred_depth=self.image_depth_generation(rays_o,rays_d,outputs["sigma"],outputs["rgbs"])
    #     # pred_rgb=results["image"]
    #     del rays_o,rays_d,outputs
    #     # pred_depth=results["depth"]
    #     pred_rgb = pred_rgb.reshape(-1, H, W, 3)
    #     pred_depth = pred_depth.reshape(-1, H, W)
    #     return pred_rgb, pred_depth
    def test_step(self, data, bg_color=None, perturb=False):
        rays_o = data['rays_o']
        rays_d = data['rays_d']
        fixed_rays_o = data.get("fixed_rays_o",None)
        fixed_rays_d = data.get("fixed_rays_d",None)
    
        H, W = data['H'], data['W']
        # print(H,W)
        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        # Process rays directly without pre-computed values
        pred_rgb, pred_depth = self.render_with_blending_fast(
            rays_o, rays_d,
            fixed_rays_d=fixed_rays_d,
            fixed_rays_o=fixed_rays_o,
            bg_color=bg_color,
            perturb=perturb
        )

        # pred_rgb, pred_depth = self.image_depth_generation(
        #     rays_o, rays_d,
        #     fixed_rays_d=fixed_rays_d,
        #     fixed_rays_o=fixed_rays_o,
        #     bg_color=bg_color,
        #     perturb=perturb
        # )


        del rays_o, rays_d
        pred_rgb = pred_rgb.reshape(-1, H, W, 3)
        pred_depth = pred_depth.reshape(-1, H, W)
        return pred_rgb, pred_depth
    
    def test_gui(self, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1):
        """GUI test method for combined scene"""
        # Use active trainer's get_rays functionality
        # downscale=1
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale
        # breakpoint()
        
        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)
        # print
        rays = get_rays(pose, intrinsics, rH, rW, -1)
        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
        }
        if len(self.scene_editing["static_objects"])>0 and self.scene_editing["fixed_pose"] is None:
            print("Editing Scene",pose)
            # breakpoint()
            self.scene_editing["fixed_pose"]=pose

        if len(self.scene_editing["static_objects"])==0:
            self.scene_editing["fixed_pose"]=None

        if not self.scene_editing["fixed_pose"] is None:
            random_pose=self.scene_editing["fixed_pose"]
            # print("Editing Scene",pose,random_pose)
            # breakpoint()
            # random_pose[:,:3,3]=torch.rand(1,3)
            fixed_rays=get_rays(random_pose, intrinsics, rH, rW, -1)
            # print("Editing Scene",self.scene_editing["fixed_pose"])
            data.update({
                "fixed_rays_o":fixed_rays["rays_o"],
                "fixed_rays_d":fixed_rays["rays_d"],
            })
            # print(data.keys())
        
        # print(rays['rays_o'].shape)
        # Set all models to eval mode
        for key,trainer in self.trainers.items():
            trainer.model.eval()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.active_trainer.fp16):
                preds, preds_depth = self.test_step(
                    data,
                    bg_color=bg_color,
                    # perturb=False if spp == 1 else spp
                    perturb=False
                )

        # if self.ema is not None:
        #     self.ema.restore()
        # breakpoint()
        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        if self.opt.color_space == 'linear':
            preds = linear_to_srgb(preds)


        pred_depth = preds_depth.detach().clone().cpu().numpy()
        preds = preds.detach().clone().cpu().numpy()


        return {
            'image': preds,
            'depth': pred_depth
        }

    def train_gui(self, train_loader, step=16):
        """GUI training method - trains each model independently"""
        outputs = {}
        for i, trainer in enumerate(self.trainers):
            trainer_output = trainer.train_gui(train_loader, step)
            outputs[f'model_{i}'] = trainer_output
        return outputs

    def save_checkpoint(self, name=None):
        """Save checkpoints for all trainers"""
        if name is None:
            name = f'multi_nerf_ep{self.active_trainer.epoch:04d}'
        
        for i, trainer in enumerate(self.trainers):
            trainer.save_checkpoint(f"{name}_model{i}")

    def load_checkpoint(self, checkpoint_list):
        """Load checkpoints for all trainers"""
        if len(checkpoint_list) != len(self.trainers):
            raise ValueError("Number of checkpoints must match number of trainers")
        
        for trainer, checkpoint in zip(self.trainers, checkpoint_list):
            trainer.load_checkpoint(checkpoint)

    def evaluate_one_epoch(self, loader):
        """Evaluate combined scene"""
        self.active_trainer.log(f"++> Evaluating Multi-NeRF...")

        # Set all models to eval mode
        for trainer in self.trainers:
            trainer.model.eval()

        total_psnr = 0
        total_samples = 0

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt}')

        with torch.no_grad():
            for data in loader:
                pred_rgb, pred_depth = self.test_step(data)
                
                # Compute metrics if ground truth is available
                if 'images' in data:
                    gt_rgb = data['images'].to(self.device)
                    psnr = -10 * torch.log10(torch.mean((pred_rgb - gt_rgb) ** 2))
                    total_psnr += psnr.item()
                    total_samples += 1

                pbar.update(loader.batch_size)

        pbar.close()
        
        if total_samples > 0:
            avg_psnr = total_psnr / total_samples
            self.active_trainer.log(f"===> Average PSNR: {avg_psnr:.6f}")

        return avg_psnr if total_samples > 0 else None
    
def load_nerf_gui(workspace,path="dummy", ckpt='latest', W=600, H=600, radius=5, fovy=50, max_spp=64):
    # Initialize configuration
    opt = argparse.Namespace()
    
    # Required settings
    opt.path = "/dumm"
    opt.workspace = workspace
    opt.ckpt = ckpt
    opt.W = W
    opt.H = H
    opt.radius = radius
    opt.fovy = fovy
    opt.max_spp = max_spp
    
    # Default settings that were originally command-line arguments
    opt.O = True  # Enable optimized mode
    opt.fp16 = True
    opt.cuda_ray = True
    opt.preload = True
    opt.test = True  # We're in test mode when loading GUI
    
    # Dataset options
    opt.bound = 2
    opt.scale = 0.33
    opt.offset = [0, 0, 0]
    opt.dt_gamma = 1/128
    opt.min_near = 0.2
    opt.density_thresh = 10
    opt.bg_radius = -1
    
    # Training options (needed by trainer)
    opt.iters = 30000
    opt.lr = 1e-2
    opt.num_rays = 4096
    opt.num_steps = 128
    opt.upsample_steps = 0
    opt.update_extra_interval = 16
    opt.max_ray_batch = 4096
    opt.patch_size = 1
    opt.max_steps = 1024
    opt.color_space = 'srgb'
    
    # Other required options
    opt.ff = False
    opt.tcnn = False
    opt.error_map = False
    opt.clip_text = ''
    opt.rand_pose = -1
    opt.gui = True
    opt.seed = 0
    
    # Initialize model
    from nerf.network import NeRFNetwork
    model = NeRFNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
    )
    
    # Setup device and criterion
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.MSELoss(reduction='none')
    
    # Initialize metrics and trainer
    metrics = [PSNRMeter(), LPIPSMeter(device=device)]
    # paths=[]
    opt.workspace = "/home/kojogyaase/Projects/Research/torch-ngp/lego_nerf"
    trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, 
                     criterion=criterion, fp16=opt.fp16, metrics=metrics, 
                     use_checkpoint=opt.ckpt)
    opt2=copy.deepcopy(opt)
    # opt2.workspace="/home/kojogyaase/Projects/Research/torch-ngp/trial_nerf"
    # opt.workspace = "/home/kojogyaase/Projects/Research/torch-ngp/laptop_nerf"
    # model2 = NeRFNetwork(
    #     encoding="hashgrid",
    #     bound=opt.bound,
    #     cuda_ray=opt.cuda_ray,
    #     density_scale=1,
    #     min_near=opt.min_near,
    #     density_thresh=opt.density_thresh,
    #     bg_radius=opt.bg_radius,
    # )
    # time.sleep(2.0)
    trainer2 = Trainer('ngp', opt2, model, device=device, workspace=opt2.workspace, 
                     criterion=criterion, fp16=opt2.fp16, metrics=metrics, 
                     use_checkpoint=opt2.ckpt)
    
    # trainer3 = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, 
    #                  criterion=criterion, fp16=opt.fp16, metrics=metrics, 
    #                  use_checkpoint=opt.ckpt)
    # breakpoint()
    # multi_trainer=MultiNeRFTrainer({"base":trainer})
    multi_trainer=MultiNeRFTrainer({
                                    "base":trainer,
                                    "base_2":trainer2,
                                    # "base_3":trainer3
                                    })
    
    # Initialize GUI
    gui = NeRFGUI(opt, multi_trainer)
    return gui

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_nerf_data>")
        sys.exit(1)
        
    path = sys.argv[1]
    gui = load_nerf_gui(path)
    gui.render()