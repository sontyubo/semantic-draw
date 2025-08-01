# Copyright (c) 2025 Jaerin Lee

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from collections import deque
from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from diffusers import AutoencoderTiny
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.pipelines.stable_diffusion_3 import StableDiffusion3Pipeline
from einops import rearrange
from peft import PeftModel
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from data import BackgroundObject, BackgroundState, LayerObject  #, LayerState
from util import gaussian_lowpass, shift_to_mask_bbox_center


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class SemanticDraw3(nn.Module):
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        sd_version: Literal['3'] = '3',
        hf_key: Optional[str] = None,
        lora_key: Optional[str] = None,
        use_tiny_vae: bool = True,
        t_index_list: List[int] = [0, 4, 12, 25, 37], # [0, 5, 16, 18, 20, 37], Magic number.
        width: int = 512,
        height: int = 512,
        frame_buffer_size: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 0.0,
        delta: float = 1.0,
        cfg_type: Literal['none', 'full', 'self', 'initialize'] = 'none',
        seed: int = 2024,
        seedfix: bool = False,
        autoflush: bool = True,
        default_mask_std: float = 8.0,
        default_mask_strength: float = 1.0,
        default_prompt_strength: float = 0.95,
        bootstrap_steps: int = 1,
        bootstrap_mix_steps: float = 1.0,
        # bootstrap_leak_sensitivity: float = 0.2,
        preprocess_mask_cover_alpha: float = 0.1, # TODO
        prompt_queue_capacity: int = 256,
        mask_type: Literal['discrete', 'semi-continuous', 'continuous'] = 'continuous',
        lora_weight: float = 1.0,
        i2t_to_device: bool = False,
        use_xformers: bool = True,
    ) -> None:
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.seed = seed
        self.seedfix = seedfix
        self.sd_version = sd_version

        self.autoflush = autoflush
        self.default_mask_std = default_mask_std
        self.default_mask_strength = default_mask_strength
        self.default_prompt_strength = default_prompt_strength
        self.bootstrap_steps = (
            bootstrap_steps > torch.arange(len(t_index_list))).to(dtype=self.dtype, device=self.device)
        self.bootstrap_mix_steps = bootstrap_mix_steps
        self.bootstrap_mix_ratios = (
            bootstrap_mix_steps - torch.arange(len(t_index_list), dtype=self.dtype, device=self.device)).clip_(0, 1)
        # self.bootstrap_leak_sensitivity = bootstrap_leak_sensitivity
        self.preprocess_mask_cover_alpha = preprocess_mask_cover_alpha
        self.mask_type = mask_type

        self.guidance_rescale = 0.7

        ### State definition

        # [0. Start]                 -(prepare)->           [1. Initialized]
        # [1. Initialized]           -(update_background)-> [2. Background Registered] (len(self.prompts)==0)
        # [2. Background Registered] -(update_layers)->     [3. Unflushed] (len(self.prompts)>0)

        # [3. Unflushed]             -(flush)->             [4. Ready]
        # [4. Ready]                 -(any updates)->       [3. Unflushed]
        # [4. Ready]                 -(__call__)->          [4. Ready], continuously returns generated image.

        self.ready_checklist = {
            'initialized': False,
            'background_registered': False,
            'layers_ready': False,
            'flushed': False,
        }

        ### Session state update queue: for lazy update policy for streaming applications.

        self.update_buffer = {
            'background': None,                            # Maintains a single instance of BackgroundObject
            'layers': deque(maxlen=prompt_queue_capacity), # Maintains a queue of LayerObjects
        }

        print('[INFO]     Loading Stable Diffusion...')
        get_scheduler = lambda pipe: FlashFlowMatchEulerDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", subfolder="scheduler")

        if hf_key is not None:
            print(f'[INFO] Using Hugging Face custom model key: {hf_key}')
        else:
            hf_key = "stabilityai/stable-diffusion-3-medium-diffusers"

        transformer = SD3Transformer2DModel.from_pretrained(
            hf_key,
            subfolder="transformer",
            torch_dtype=torch.float16,
        ).to(self.device)

        transformer = PeftModel.from_pretrained(transformer, "jasperai/flash-sd3").to(self.device)

        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            transformer=transformer,
            torch_dtype=torch.float16,
            text_encoder_3=None,
            tokenizer_3=None
        ).to(self.device)

        ### Internally stored "Session" states

        self.state = {
            'background': BackgroundState(), # Maintains a single instance of BackgroundState
            # 'layers': LayerState(),          # Maintains a single instance of LayerState
            'model_key': hf_key,          # The Hugging Face model ID.
        }

        # Create model
        self.i2t_processor = Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b')
        self.i2t_model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b')
        if i2t_to_device:
            self.i2t_processor = self.i2t_processor.to(device)
            self.i2t_model = self.self.i2t_model.to(device)

        self.vae = (
            AutoencoderTiny.from_pretrained('madebyollin/taesd3').to(device=self.device, dtype=self.dtype)
            if use_tiny_vae else self.pipe.vae
        )
        # self.tokenizer = self.pipe.tokenizer
        # self.tokenizer_2 = self.pipe.tokenizer_2 if hasattr(self.pipe, 'tokenizer_2') else None
        self.text_encoder = self.pipe.text_encoder
        self.text_encoder_2 = self.pipe.text_encoder_2 if hasattr(self.pipe, 'text_encoder_2') else None
        self.text_encoder_3 = self.pipe.text_encoder_3 if hasattr(self.pipe, 'text_encoder_3') else None
        self.transformer = self.pipe.transformer
        self.vae_scale_factor = self.pipe.vae_scale_factor
        self.joint_attention_kwargs = None
        self.pipe._joint_attention_kwargs = None

        self.scheduler = get_scheduler(self.pipe)
        self.scheduler.set_timesteps(num_inference_steps)

        self.generator = None

        # Lock the canvas size--changing the canvas size can be implemented by reloading the module.
        self.height = height
        self.width = width
        self.latent_height = int(height // self.pipe.vae_scale_factor)
        self.latent_width = int(width // self.pipe.vae_scale_factor)

        # For bootstrapping.
        self.white = self.encode_imgs(torch.ones(1, 3, height, width, dtype=self.dtype, device=self.device))

        # StreamDiffusion setting.
        self.t_list = t_index_list
        assert len(self.t_list) > 1, 'Current version only supports diffusion models with multiple steps.'
        self.frame_bff_size = frame_buffer_size  # f
        self.denoising_steps_num = len(self.t_list)  # t=2
        self.cfg_type = cfg_type
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = 1.0 if self.cfg_type == 'none' else guidance_scale
        self.delta = delta

        self.batch_size = self.denoising_steps_num * frame_buffer_size  # T = t*f
        if self.cfg_type == 'initialize':
            self.trt_transformer_batch_size = (self.denoising_steps_num + 1) * self.frame_bff_size
        elif self.cfg_type == 'full':
            self.trt_transformer_batch_size = 2 * self.denoising_steps_num * self.frame_bff_size
        else:
            self.trt_transformer_batch_size = self.denoising_steps_num * frame_buffer_size

        print('[INFO]     Model is loaded!')

        self.reset_seed(self.generator, seed)
        self.reset_latent()
        self.prepare()

        print('[INFO]     Parameters prepared!')

        self.ready_checklist['initialized'] = True

    @property
    def background(self) -> BackgroundState:
        return self.state['background']

    # @property
    # def layers(self) -> LayerState:
    #     return self.state['layers']

    @property
    def num_layers(self) -> int:
        return len(self.prompts) if hasattr(self, 'prompts') else 0

    @property
    def is_ready_except_flush(self) -> bool:
        return all(v for k, v in self.ready_checklist.items() if k != 'flushed')

    @property
    def is_flush_needed(self) -> bool:
        return self.autoflush and not self.ready_checklist['flushed']

    @property
    def is_ready(self) -> bool:
        return self.is_ready_except_flush and not self.is_flush_needed

    @property
    def is_dirty(self) -> bool:
        return not (self.update_buffer['background'] is None and len(self.update_buffer['layers']) == 0)

    @property
    def has_background(self) -> bool:
        return self.background.is_empty

    # @property
    # def has_layers(self) -> bool:
    #     return len(self.layers) > 0

    def __repr__(self) -> str:
        return (
            f'{type(self).__name__}(\n\tbackground: {str(self.background)},\n\t'
            f'model_key: {self.state["model_key"]}\n)'
            # f'layers: {str(self.layers)},\n\tmodel_key: {self.state["model_key"]}\n)'
        )

    def check_integrity(self, throw_error: bool = True) -> bool:
        p = len(self.prompts)
        flag = (
            p != len(self.negative_prompts) or
            p != len(self.prompt_strengths) or
            p != len(self.masks) or
            p != len(self.mask_strengths) or
            p != len(self.mask_stds) or
            p != len(self.original_masks)
        )
        if flag and throw_error:
            print(
                f'LayerState(\n\tlen(prompts): {p},\n\tlen(negative_prompts): {len(self.negative_prompts)},\n\t'
                f'len(prompt_strengths): {len(self.prompt_strengths)},\n\tlen(masks): {len(self.masks)},\n\t'
                f'len(mask_stds): {len(self.mask_stds)},\n\tlen(mask_strengths): {len(self.mask_strengths)},\n\t'
                f'len(original_masks): {len(self.original_masks)}\n)'
            )
            raise ValueError('[ERROR]    LayerState is corrupted!')
        return not flag

    def check_ready(self) -> bool:
        all_except_flushed = all(v for k, v in self.ready_checklist.items() if k != 'flushed')
        if all_except_flushed:
            if self.is_flush_needed:
                self.flush()
            return True

        print('[WARNING]  MagicDraw module is not ready yet! Complete the checklist:')
        for k, v in self.ready_checklist.items():
            prefix = '  [ v ] ' if v else '  [ x ] '
            print(prefix + k.replace('_', ' '))
        return False

    def reset_seed(self, generator: Optional[torch.Generator] = None, seed: Optional[int] = None) -> None:
        generator = torch.Generator(self.device) if generator is None else generator
        seed = self.seed if seed is None else seed
        self.generator = generator
        self.generator.manual_seed(seed)

        self.init_noise = torch.randn((self.batch_size, self.transformer.config.in_channels, self.latent_height, self.latent_width),
            generator=generator, device=self.device, dtype=self.dtype)
        self.stock_noise = torch.zeros_like(self.init_noise)

        self.ready_checklist['flushed'] = False

    def reset_latent(self) -> None:
        # initialize x_t_latent (it can be any random tensor)
        b = (self.denoising_steps_num - 1) * self.frame_bff_size
        self.x_t_latent_buffer = torch.zeros(
            (b, self.transformer.config.in_channels, self.latent_height, self.latent_width), dtype=self.dtype, device=self.device)

    def reset_state(self) -> None:
        # TODO Reset states for context switch between multiple users.
        pass

    def prepare_flashflowmatch(self):
        r"""Set up different inference schedule for the diffusion model.

        You do not have to run this explicitly if you want to use the default
        setting, but if you want other time schedules, run this function
        between the module initialization and the main call.

        Note:
          - Recommended t_index_lists for LCMs:
              - [0, 12, 25, 37]: Default schedule for 4 steps. Best for
                  panorama. Not recommended if you want to use bootstrapping.
                  Because bootstrapping stage affects the initial structuring
                  of the generated image & in this four step LCM, this is done
                  with only at the first step, the structure may be distorted.
              - [0, 4, 12, 25, 37]: Recommended if you would use 1-step boot-
                  strapping. Default initialization in this implementation.
              - [0, 5, 16, 18, 20, 37]: Recommended if you would use 2-step
                  bootstrapping.
          - Due to the characteristic of SD1.5 LCM LoRA, setting
            `num_inference_steps` larger than 20 may results in overly blurry
            and unrealistic images. Beware!
        """
        # make sub timesteps list based on the indices in the t_list list and the values in the timesteps list
        self.timesteps = self.scheduler.timesteps.to(self.device)
        self.sub_timesteps = []
        for t in self.t_list:
            self.sub_timesteps.append(self.timesteps[t])
        sub_timesteps_tensor = torch.tensor(self.sub_timesteps, dtype=self.dtype, device=self.device)
        self.sub_timesteps_tensor = sub_timesteps_tensor.repeat_interleave(self.frame_bff_size, dim=0)

        # FlashFlowMatchEulerDiscreteScheduler
        # https://github.com/initml/diffusers/blob/clement/feature/flash_sd3/src/diffusers/schedulers/scheduling_flash_flow_match_euler_discrete.py

        self.sigmas = self.scheduler.sigmas[torch.tensor(self.t_list)].to(self.device)
        self.sigmas_next = torch.cat([self.sigmas, self.sigmas.new_zeros(1)])[1:].to(self.device)

        self.sigmas = self.sigmas.repeat_interleave(self.frame_bff_size, dim=0)
        self.sigmas_next = self.sigmas_next.repeat_interleave(self.frame_bff_size, dim=0)

        noise_lvs = self.sigmas * (self.sigmas**2 + 1)**(-0.5)
        self.noise_lvs = noise_lvs[None, :, None, None, None]
        self.next_noise_lvs = torch.cat([noise_lvs[1:], noise_lvs.new_zeros(1)])[None, :, None, None, None]

    def prepare(self) -> None:
        self.prepare_flashflowmatch()

    @torch.no_grad()
    def get_text_prompts(self, image: Image.Image) -> str:
        r"""A convenient method to extract text prompt from an image.

        This is called if the user does not provide background prompt but only
        the background image. We use BLIP-2 to automatically generate prompts.

        Args:
            image (Image.Image): A PIL image.

        Returns:
            A single string of text prompt.
        """
        question = 'Question: What are in the image? Answer:'
        inputs = self.i2t_processor(image, question, return_tensors='pt')
        out = self.i2t_model.generate(**inputs) #, max_new_tokens=77)
        prompt = self.i2t_processor.decode(out[0], skip_special_tokens=True).strip()
        return prompt

    @torch.no_grad()
    def encode_imgs(
        self,
        imgs: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        add_noise: bool = False,
    ) -> torch.Tensor:
        r"""A wrapper function for VAE encoder of the latent diffusion model.

        Args:
            imgs (torch.Tensor): An image to get StableDiffusion latents.
                Expected shape: (B, 3, H, W). Expected pixel scale: [0, 1].
            generator (Optional[torch.Generator]): Seed for KL-Autoencoder.
            add_noise (bool): Turn this on for a noisy latent.

        Returns:
            An image latent embedding with 1/8 size (depending on the auto-
            encoder. Shape: (B, 4, H//8, W//8).
        """
        def _retrieve_latents(
            encoder_output: torch.Tensor,
            generator: Optional[torch.Generator] = None,
            sample_mode: str = 'sample',
        ):
            if hasattr(encoder_output, 'latent_dist') and sample_mode == 'sample':
                return encoder_output.latent_dist.sample(generator)
            elif hasattr(encoder_output, 'latent_dist') and sample_mode == 'argmax':
                return encoder_output.latent_dist.mode()
            elif hasattr(encoder_output, 'latents'):
                return encoder_output.latents
            else:
                raise AttributeError('[ERROR]    Could not access latents of provided encoder_output')

        imgs = 2 * imgs - 1
        latents = self.vae.config.scaling_factor * _retrieve_latents(self.vae.encode(imgs), generator=generator)
        if add_noise:
            latents = self.alpha_prod_t_sqrt[0] * latents + self.beta_prod_t_sqrt[0] * self.init_noise[0]
        return latents

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        r"""A wrapper function for VAE decoder of the latent diffusion model.

        Args:
            latents (torch.Tensor): An image latent to get associated images.
                Expected shape: (B, 4, H//8, W//8).

        Returns:
            An image latent embedding with 1/8 size (depending on the auto-
            encoder. Shape: (B, 3, H, W).
        """
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        return image.clip_(-1, 1) * 0.5 + 0.5

    @torch.no_grad()
    def update_background(
        self,
        image: Optional[Image.Image] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
    ) -> bool:
        flag_changed = False
        if image is not None:
            image_ = image.resize((self.width, self.height))
            prompt = self.get_text_prompts(image_) if prompt is None else prompt
            negative_prompt = '' if negative_prompt is None else negative_prompt
            embed = self.pipe.encode_prompt(
                prompt=[prompt],
                prompt_2=None,
                prompt_3=None,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=(self.guidance_scale > 1.0),
                negative_prompt=[negative_prompt],
            )  # ((1, 77, 768): cond, (1, 77, 768): uncond)

            self.state['background'].image = image
            self.state['background'].latent = (
                self.encode_imgs(T.ToTensor()(image_)[None].to(self.device, self.dtype))
            )  # (1, 3, H, W)
            self.state['background'].prompt = prompt
            self.state['background'].negative_prompt = negative_prompt
            self.state['background'].embed = embed

            if self.bootstrap_steps[0] > 0:
                mix_ratio = self.bootstrap_mix_ratios[:, None, None, None]
                self.bootstrap_latent = mix_ratio * self.white + (1.0 - mix_ratio) * self.state['background'].latent

            self.ready_checklist['background_registered'] = True
            flag_changed = True
        else:
            if not self.ready_checklist['background_registered']:
                print('[WARNING]  Register background image first! Request ignored.')
                return False

            if prompt is not None:
                self.background.prompt = prompt
                flag_changed = True
            if negative_prompt is not None:
                self.background.negative_prompt = negative_prompt
                flag_changed = True
            if flag_changed:
                self.background.embed = self.pipe.encode_prompt(
                    prompt=[self.background.prompt],
                    prompt_2=None,
                    prompt_3=None,
                    device=self.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=(self.guidance_scale > 1.0),
                    negative_prompt=[self.background.negative_prompt],
                )  # ((1, 77, 768): cond, (1, 77, 768): uncond)
    
        self.ready_checklist['flushed'] = not flag_changed
        return flag_changed

    @torch.no_grad()
    def process_mask(
        self,
        masks: Optional[Union[torch.Tensor, Image.Image, List[Image.Image]]] = None,
        strength: Optional[Union[torch.Tensor, float]] = None,
        std: Optional[Union[torch.Tensor, float]] = None,
    ) -> Tuple[torch.Tensor]:
        r"""Fast preprocess of masks for region-based generation with fine-
        grained controls.

        Mask preprocessing is done in four steps:
         1. Resizing: Resize the masks into the specified width and height by
            nearest neighbor interpolation.
         2. (Optional) Ordering: Masks with higher indices are considered to
            cover the masks with smaller indices. Covered masks are decayed
            in its alpha value by the specified factor of
            `preprocess_mask_cover_alpha`.
         3. Blurring: Gaussian blur is applied to the mask with the specified
            standard deviation (isotropic). This results in gradual increase of
            masked region as the timesteps evolve, naturally blending fore-
            ground and the predesignated background. Not strictly required if
            you want to produce images from scratch withoout background.
         4. Quantization: Split the real-numbered masks of value between [0, 1]
            into predefined noise levels for each quantized scheduling step of
            the diffusion sampler. For example, if the diffusion model sampler
            has noise level of [0.9977, 0.9912, 0.9735, 0.8499, 0.5840], which
            is the default noise level of this module with schedule [0, 4, 12,
            25, 37], the masks are split into binary masks whose values are
            greater than these levels. This results in tradual increase of mask
            region as the timesteps increase. Details are described in our
            paper.

        On the Three Modes of `mask_type`:
            `self.mask_type` is predefined at the initialization stage of this
            pipeline. Three possible modes are available: 'discrete', 'semi-
            continuous', and 'continuous'. These define the mask quantization
            modes we use. Basically, this (subtly) controls the smoothness of
            foreground-background blending. Continuous modes produces nonbinary
            masks to further blend foreground and background latents by linear-
            ly interpolating between them. Semi-continuous masks only applies
            continuous mask at the last step of the LCM sampler. Due to the
            large step size of the LCM scheduler, we find that our continuous
            blending helps generating seamless inpainting and editing results.

        Args:
            masks (Union[torch.Tensor, Image.Image, List[Image.Image]]): Masks.
            strength (Optional[Union[torch.Tensor, float]]): Mask strength that
                overrides the default value. A globally multiplied factor to
                the mask at the initial stage of processing. Can be applied
                seperately for each mask.
            std (Optional[Union[torch.Tensor, float]]): Mask blurring Gaussian
                kernel's standard deviation. Overrides the default value. Can
                be applied seperately for each mask.

        Returns: A tuple of tensors.
          - masks: Preprocessed (ordered, blurred, and quantized) binary/non-
                binary masks (see the explanation on `mask_type` above) for
                region-based image synthesis.
          - strengths: Return mask strengths for caching.
          - std: Return mask blur standard deviations for caching.
          - original_masks: Return original masks for caching.
        """
        if masks is None:
            kwargs = {'dtype': self.dtype, 'device': self.device}
            original_masks = torch.zeros((0, 1, self.latent_height, self.latent_width), dtype=self.dtype)
            masks = torch.zeros((0, self.batch_size, 1, self.latent_height, self.latent_width), **kwargs)
            strength = torch.zeros((0,), **kwargs)
            std = torch.zeros((0,), **kwargs)
            return masks, strength, std, original_masks

        if isinstance(masks, Image.Image):
            masks = [masks]
        if isinstance(masks, (tuple, list)):
            # Assumes white background for Image.Image;
            # inverted boolean masks with shape (1, 1, H, W) for torch.Tensor.
            masks = torch.cat([
                # (T.ToTensor()(mask.resize((self.width, self.height), Image.NEAREST)) < 0.5)[None, :1]
                (1.0 - T.ToTensor()(mask.resize((self.width, self.height), Image.BILINEAR)))[None, :1]
                for mask in masks
            ], dim=0).float().clip_(0, 1)
        original_masks = masks            
        masks = masks.float().to(self.device)

        # Background mask alpha is decayed by the specified factor where foreground masks covers it.
        if self.preprocess_mask_cover_alpha > 0:
            masks = torch.stack([
                torch.where(
                    masks[i + 1:].sum(dim=0) > 0,
                    mask * self.preprocess_mask_cover_alpha,
                    mask,
                ) if i < len(masks) - 1 else mask
                for i, mask in enumerate(masks)
            ], dim=0)

        if std is None:
            std = self.default_mask_std
        if isinstance(std, (int, float)):
            std = [std] * len(masks)
        if isinstance(std, (list, tuple)):
            std = torch.as_tensor(std, dtype=torch.float, device=self.device)

        # Mask preprocessing parameters are fetched from the default settings.
        if strength is None:
            strength = self.default_mask_strength
        if isinstance(strength, (int, float)):
            strength = [strength] * len(masks)
        if isinstance(strength, (list, tuple)):
            strength = torch.as_tensor(strength, dtype=torch.float, device=self.device)

        if (std > 0).any():
            std = torch.where(std > 0, std, 1e-5)
            masks = gaussian_lowpass(masks, std)
        # NOTE: This `strength` aligns with `denoising strength`. However, with LCM, using strength < 0.96
        #       gives unpleasant results.
        masks = masks * strength[:, None, None, None]
        masks = masks.unsqueeze(1).repeat(1, self.noise_lvs.shape[1], 1, 1, 1)

        if self.mask_type == 'discrete':
            # Discrete mode.
            masks = masks > self.noise_lvs
        elif self.mask_type == 'semi-continuous':
            # Semi-continuous mode (continuous at the last step only).
            masks = torch.cat((
                masks[:, :-1] > self.noise_lvs[:, :-1],
                (
                    (masks[:, -1:] - self.next_noise_lvs[:, -1:])
                    / (self.noise_lvs[:, -1:] - self.next_noise_lvs[:, -1:])
                ).clip_(0, 1),
            ), dim=1)
        elif self.mask_type == 'continuous':
            # Continuous mode: Have the exact same `1` coverage with discrete mode, but the mask gradually
            #                  decreases continuously after the discrete mode boundary to become `0` at the
            #                  next lower threshold.
            masks = ((masks - self.next_noise_lvs) / (self.noise_lvs - self.next_noise_lvs)).clip_(0, 1)

        # NOTE: Post processing mask strength does not align with conventional 'denoising_strength'. However,
        #       fine-grained mask alpha channel tuning is available with this form.
        # masks = masks * strength[None, :, None, None, None]

        masks = rearrange(masks.float(), 'p t () h w -> (p t) () h w')
        masks = F.interpolate(masks, size=(self.latent_height, self.latent_width), mode='nearest')
        masks = rearrange(masks.to(self.dtype), '(p t) () h w -> p t () h w', p=len(std))
        return masks, strength, std, original_masks

    @torch.no_grad()
    def update_layers(
        self,
        prompts: Union[str, List[str]],
        negative_prompts: Optional[Union[str, List[str]]] = None,
        suffix: Optional[str] = None, #', background is ',
        prompt_strengths: Optional[Union[torch.Tensor, float, List[float]]] = None,
        masks: Optional[Union[torch.Tensor, Image.Image, List[Image.Image]]] = None,
        mask_strengths: Optional[Union[torch.Tensor, float, List[float]]] = None,
        mask_stds: Optional[Union[torch.Tensor, float, List[float]]] = None,
    ) -> None:
        if not self.ready_checklist['background_registered']:
            print('[WARNING]  Register background image first! Request ignored.')
            return

        ### Register prompts

        if isinstance(prompts, str):
            prompts = [prompts]
        if negative_prompts is None:
            negative_prompts = ''
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]
        fg_prompt = [p + suffix + self.background.prompt if suffix is not None else p for p in prompts]
        self.prompts = fg_prompt
        self.negative_prompts = negative_prompts
        p = self.num_layers

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=fg_prompt,
            prompt_2=None,
            prompt_3=None,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=(self.guidance_scale > 1.0),
            negative_prompt=negative_prompts,
        )  # (p, 77, 768)

        if prompt_strengths is None:
            prompt_strengths = self.default_prompt_strength
        if isinstance(prompt_strengths, (int, float)):
            prompt_strengths = [prompt_strengths] * p
        if isinstance(prompt_strengths, (list, tuple)):
            prompt_strengths = torch.as_tensor(prompt_strengths, dtype=self.dtype, device=self.device)
        self.prompt_strengths = prompt_strengths

        s = prompt_strengths[:, None, None]
        self.prompt_embeds = torch.lerp(
            self.background.embed[0], prompt_embeds, s).repeat(self.batch_size, 1, 1)  # (T * p, 77, 768)

        # SD3 pipeline settings.

        if self.guidance_scale > 1.0 and self.cfg_type in ('initialize', 'full'):
            b = self.batch_size if self.cfg_type == 'full' else self.frame_bff_size
            negative_prompt_embeds = torch.lerp(
                self.background.embed[1], negative_prompt_embeds, s).repeat(b, 1, 1)  # (T * p, 77, 768)
            self.prompt_embeds = torch.cat([negative_prompt_embeds, self.prompt_embeds], dim=0)  # (2 * T * p, 77, 768)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        del negative_prompt_embeds, negative_pooled_prompt_embeds

        self.prompt_embeds = self.prompt_embeds.to(self.device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(self.device).repeat(self.batch_size, 1)
        self.pooled_prompt_embeds = pooled_prompt_embeds

        self.sub_timesteps_tensor_ = self.sub_timesteps_tensor.repeat_interleave(p)  # (T * p,)
        self.init_noise_ = self.init_noise.repeat_interleave(p, dim=0)  # (T * p, 77, 768)
        self.stock_noise_ = self.stock_noise.repeat_interleave(p, dim=0)  # (T * p, 77, 768)

        # Scheduler-dependent paramter preparation.

        self.sigmas_ = self.sigmas.repeat_interleave(p, dim=0)
        self.sigmas_next_ = self.sigmas_next.repeat_interleave(p, dim=0)
        

        ### Register new masks

        if isinstance(masks, Image.Image):
            masks = [masks]
        n = len(masks) if masks is not None else 0

        # Modificiation.
        masks, mask_strengths, mask_stds, original_masks = self.process_mask(masks, mask_strengths, mask_stds)

        self.counts = masks.sum(dim=0)  # (T, 1, h, w)
        self.bg_mask = (1 - self.counts).clip_(0, 1)  # (T, 1, h, w)
        self.masks = masks  # (p, T, 1, h, w)
        self.mask_strengths = mask_strengths  # (p,)
        self.mask_stds = mask_stds  # (p,)
        self.original_masks = original_masks  # (p, 1, h, w)

        if p > n:
            # Add more masks: counts and bg_masks are not changed, but only masks are changed.
            self.masks = torch.cat((
                self.masks,
                torch.zeros(
                    (p - n, self.batch_size, 1, self.latent_height, self.latent_width),
                    dtype=self.dtype,
                    device=self.device,
                ),
            ), dim=0)
            print(f'[WARNING]  Detected more prompts ({p}) than masks ({n}). '
                  'Automatically adds blank masks for the additional prompts.')
        elif p < n:
            # Warns user to add more prompts.
            print(f'[WARNING]  Detected more masks ({n}) than prompts ({p}). '
                  'Additional masks are ignored until more prompts are provided.')

        self.ready_checklist['layers_ready'] = True
        self.ready_checklist['flushed'] = False

    @torch.no_grad()
    def update_masks(
        self,
        masks: Optional[Union[torch.Tensor, Image.Image, List[Image.Image]]] = None,
        mask_strengths: Optional[Union[torch.Tensor, float, List[float]]] = None,
        mask_stds: Optional[Union[torch.Tensor, float, List[float]]] = None,
    ) -> None:
        if not self.ready_checklist['background_registered']:
            print('[WARNING]  Register background image first! Request ignored.')
            return

        ### Register new masks

        if isinstance(masks, Image.Image):
            masks = [masks]
        p = self.num_layers
        n = len(masks) if masks is not None else 0

        # Modificiation.
        masks, mask_strengths, mask_stds, original_masks = self.process_mask(masks, mask_strengths, mask_stds)

        self.counts = masks.sum(dim=0)  # (T, 1, h, w)
        self.bg_mask = (1 - self.counts).clip_(0, 1)  # (T, 1, h, w)
        self.masks = masks  # (p, T, 1, h, w)
        self.mask_strengths = mask_strengths  # (p,)
        self.mask_stds = mask_stds  # (p,)
        self.original_masks = original_masks  # (p, 1, h, w)

        if p > n:
            # Add more masks: counts and bg_masks are not changed, but only masks are changed.
            self.masks = torch.cat((
                self.masks,
                torch.zeros(
                    (p - n, self.batch_size, 1, self.latent_height, self.latent_width),
                    dtype=self.dtype,
                    device=self.device,
                ),
            ), dim=0)
            print(f'[WARNING]  Detected more prompts ({p}) than masks ({n}). '
                  'Automatically adds blank masks for the additional prompts.')
        elif p < n:
            # Warns user to add more prompts.
            print(f'[WARNING]  Detected more masks ({n}) than prompts ({p}). '
                  'Additional masks are ignored until more prompts are provided.')

    @torch.no_grad()
    def update_single_layer(
        self,
        idx: Optional[int] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        suffix: Optional[str] = None, #', background is ',
        prompt_strength: Optional[float] = None,
        mask: Optional[Union[torch.Tensor, Image.Image]] = None,
        mask_strength: Optional[float] = None,
        mask_std: Optional[float] = None,
    ) -> None:

        ### Possible input combinations and expected behaviors

        # The module will consider a layer, a pair of (prompt, mask), to be 'active' only if a prompt
        # is registered. A blank mask will be assigned if no mask is provided for the 'active' layer.
        # The layers should be in either of ('active', 'inactive') states. 'inactive' layers will not
        # receive any input unless equipped with prompt. 'active' layers receive any input and modify
        # their states accordingly. In the actual implementation, only the 'active' layers are stored
        # and can be accessed by the fields. Values len(self.prompts) = self.num_layers is the number
        # of 'active' layers.

        # If no background is registered. The layers should be all 'inactive'.
        if not self.ready_checklist['background_registered']:
            print('[WARNING]  Register background image first! Request ignored.')
            return

        # The first layer create request should be carrying a prompt. If only mask is drawn without a
        # prompt, it just ignores the request--the user will update her request soon.
        if self.num_layers == 0:
            if prompt is not None:
                self.update_layers(
                    prompts=prompt,
                    negative_prompts=negative_prompt,
                    suffix=suffix,
                    prompt_strengths=prompt_strength,
                    masks=mask,
                    mask_strengths=mask_strength,
                    mask_stds=mask_std,
                )
            return

        # Invalid request indices -> considered as a layer add request.
        if idx is None or idx > self.num_layers or idx < 0:
            idx = self.num_layers

        # Two modes for the layer edits: 'append mode' and 'edit mode'. 'append mode' appends a new
        # layer at the end of the layers list. 'edit mode' modifies internal variables for the given
        # index. 'append mode' is defined by the request index and strictly requires a prompt input.
        is_appending = idx == self.num_layers
        if is_appending and prompt is None:
            print(f'[WARNING]  Creating a new prompt at index ({idx}) but found no prompt. Request ignored.')
            return

        ### Register prompts

        # | prompt    | neg_prompt | append mode (idx==len)  | edit mode (idx<len)  |
        # | --------- | ---------- | ----------------------- | -------------------- |
        # | given     | given      | append new prompt embed | replace prompt embed |
        # | given     | not given  | append new prompt embed | replace prompt embed |
        # | not given | given      | NOT ALLOWED             | replace prompt embed |
        # | not given | not given  | NOT ALLOWED             | do nothing           |

        # | prompt_strength | append mode (idx==len) | edit mode (idx<len)                            |
        # | --------------- | ---------------------- | ---------------------------------------------- |
        # | given           | use given strength     | use given strength                             |
        # | not given       | use default strength   | replace strength / if no existing, use default |

        p = self.num_layers

        flag_prompt_edited = (
            prompt is not None or
            negative_prompt is not None or
            prompt_strength is not None
        )

        if flag_prompt_edited:
            is_double_cond = self.guidance_scale > 1.0 and self.cfg_type in ('initialize', 'full')

            # Synchonize the internal state.

            # We have asserted that prompt is not None if the mode is 'appending'.
            if prompt is not None:
                if suffix is not None:
                    prompt = prompt + suffix + self.background.prompt
                if is_appending:
                    self.prompts.append(prompt)
                else:
                    self.prompts[idx] = prompt

            if negative_prompt is not None:
                if is_appending:
                    self.negative_prompts.append(negative_prompt)
                else:
                    self.negative_prompts[idx] = negative_prompt
            elif is_appending:
                # Make sure that negative prompts are well specified.
                self.negative_prompts.append('')

            if is_appending:
                if prompt_strength is None:
                    prompt_strength = self.default_prompt_strength
                self.prompt_strengths = torch.cat((
                    self.prompt_strengths,
                    torch.as_tensor([prompt_strength], dtype=self.dtype, device=self.device),
                ), dim=0)
            elif prompt_strength is not None:
                self.prompt_strengths[idx] = prompt_strength

            # Edit currently stored prompt embeddings.

            if is_double_cond:
                negative_prompt_embed_, prompt_embed_ = torch.chunk(self.prompt_embeds, 2, dim=0)
                negative_prompt_embed_ = rearrange(negative_prompt_embed_, '(t p) c1 c2 -> t p c1 c2', p=p)
                prompt_embed_ = rearrange(prompt_embed_, '(t p) c1 c2 -> t p c1 c2', p=p)
            else:
                negative_prompt_embed_ = None
                prompt_embed_ = rearrange(self.prompt_embeds, '(t p) c1 c2 -> t p c1 c2', p=p)
            pooled_prompt_embed_ = rearrange(self.pooled_prompt_embeds, '(t p) c -> t p c', p=p)

            (
                prompt_embed,
                negative_prompt_embed,
                pooled_prompt_embed,
                negative_pooled_prompt_embed,
            ) = self.pipe.encode_prompt(
                prompt=self.prompts[idx],
                prompt_2=None,
                prompt_3=None,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=(self.guidance_scale > 1.0),
                negative_prompt=self.negative_prompts[idx],
            )  # (p, 77, 768)

            s = self.prompt_strengths[idx]
            t = prompt_embed_.shape[0]
            prompt_embed = torch.lerp(self.background.embed[0], prompt_embed, s)[None].repeat(t, 1, 1, 1)  # (1, 77, 768)

            # SD3 pipeline settings.

            if is_double_cond:
                negative_prompt_embed = torch.lerp(self.background.embed[1], negative_prompt_embed, s)[None].repeat(t, 1, 1, 1)  # (1, 77, 768)
                pooled_prompt_embed = torch.cat([negative_pooled_prompt_embed, pooled_prompt_embed], dim=0)

            if is_appending:
                prompt_embed_ = torch.cat((prompt_embed_, prompt_embed), dim=1)
                if is_double_cond:
                    negative_prompt_embed_ = torch.cat((negative_prompt_embed_, negative_prompt_embed), dim=1)
            else:
                prompt_embed_[:, idx:(idx + 1)] = prompt_embed
                if is_double_cond:
                    negative_prompt_embed_[:, idx:(idx + 1)] = negative_prompt_embed

            self.prompt_embeds = rearrange(prompt_embed_, 't p c1 c2 -> (t p) c1 c2')
            if is_double_cond:
                negative_prompt_embeds = rearrange(negative_prompt_embed_, 't p c1 c2 -> (t p) c1 c2')
                self.prompt_embeds = torch.cat([negative_prompt_embeds, self.prompt_embeds], dim=0)  # (2 * T * p, 77, 768)

            pooled_prompt_embed = pooled_prompt_embed.to(self.device)[None].repeat(self.batch_size, 1, 1)
            if is_appending:
                pooled_prompt_embed_ = torch.cat((pooled_prompt_embed_, pooled_prompt_embed), dim=1)
            else:
                pooled_prompt_embed_[:, idx:(idx + 1)] = pooled_prompt_embed
            self.pooled_prompt_embeds = rearrange(pooled_prompt_embed_, 't p c -> (t p) c')

            del negative_prompt_embed, negative_pooled_prompt_embed

            self.ready_checklist['flushed'] = False

        if is_appending:
            p = self.num_layers
            self.sub_timesteps_tensor_ = self.sub_timesteps_tensor.repeat_interleave(p)  # (T * p,)
            self.init_noise_ = self.init_noise.repeat_interleave(p, dim=0)  # (T * p, 77, 768)
            self.stock_noise_ = self.stock_noise.repeat_interleave(p, dim=0)  # (T * p, 77, 768)

            # Scheduler-dependent paramter preparation.

            self.sigmas_ = self.sigmas.repeat_interleave(p, dim=0)
            self.sigma_next_ = self.sigma_next.repeat_interleave(p, dim=0)

        ### Register new masks

        # | mask      | std / str | append mode (idx==len)       | edit mode (idx<len)           |
        # | --------- | --------- | ---------------------------- | ----------------------------- |
        # | given     | given     | create mask with given val   | create mask with given val    |
        # | given     | not given | create mask with default val | create mask with existing val |
        # | not given | given     | create blank mask            | replace mask with given val   |
        # | not given | not given | create blank mask            | do nothing                    |

        flag_nonzero_mask = False
        if mask is not None:
            # Mask image is given -> create mask.
            mask, strength, std, original_mask = self.process_mask(mask, mask_strength, mask_std)
            flag_nonzero_mask = True

        elif is_appending:
            # No given mask & append mode  -> create white mask.
            mask = torch.zeros(
                (1, self.batch_size, 1, self.latent_height, self.latent_width),
                dtype=self.dtype,
                device=self.device,
            )
            strength = torch.as_tensor([self.default_mask_strength], dtype=self.dtype, device=self.device)
            std = torch.as_tensor([self.default_mask_std], dtype=self.dtype, device=self.device)
            original_mask = torch.zeros((1, 1, self.height, self.width), dtype=self.dtype, device=self.device)

        elif mask_std is not None or mask_strength is not None:
            # No given mask & edit mode & given std / str -> replace existing mask with given std / str.
            if mask_std is None:
                mask_std = self.mask_stds[idx:(idx + 1)]
            if mask_strength is None:
                mask_strength = self.mask_strengths[idx:(idx + 1)]
            mask, strength, std, original_mask = self.process_mask(
                self.original_masks[idx:(idx + 1)], mask_strength, mask_std)
            flag_nonzero_mask = True

        else:
            # No given mask & no given std & edit mode -> Do nothing.
            return

        if is_appending:
            # Append mode.
            self.masks = torch.cat((self.masks, mask), dim=0)  # (p, T, 1, h, w)
            self.mask_strengths = torch.cat((self.mask_strengths, strength), dim=0)  # (p,)
            self.mask_stds = torch.cat((self.mask_stds, std), dim=0)  # (p,)
            self.original_masks = torch.cat((self.original_masks, original_mask), dim=0)  # (p, 1, h, w)
            if flag_nonzero_mask:
                self.counts = self.counts + mask[0] if hasattr(self, 'counts') else mask[0]  # (T, 1, h, w)
                self.bg_mask = (1 - self.counts).clip_(0, 1)  # (T, 1, h, w)
        else:
            # Edit mode.
            if flag_nonzero_mask:
                self.counts = self.counts - self.masks[idx] + mask[0]  # (T, 1, h, w)
                self.bg_mask = (1 - self.counts).clip_(0, 1)  # (T, 1, h, w) 
            self.masks[idx:(idx + 1)] = mask  # (p, T, 1, h, w)
            self.mask_strengths[idx:(idx + 1)] = strength  # (p,)
            self.mask_stds[idx:(idx + 1)] = std  # (p,)
            self.original_masks[idx:(idx + 1)] = original_mask  # (p, 1, h, w)

        # if flag_nonzero_mask:
        #     self.ready_checklist['flushed'] = False

    @torch.no_grad()
    def register_all(
        self,
        prompts: Union[str, List[str]],
        masks: Union[Image.Image, List[Image.Image]],
        background: Image.Image,
        background_prompt: Optional[str] = None,
        background_negative_prompt: str = '',
        negative_prompts: Union[str, List[str]] = '',
        suffix: Optional[str] = None, #', background is ',
        prompt_strengths: float = 1.0,
        mask_strengths: float = 1.0,
        mask_stds: Union[torch.Tensor, float] = 10.0,
    ) -> None:
        # The order of this registration should not be changed!
        self.update_background(background, background_prompt, background_negative_prompt)
        self.update_layers(prompts, negative_prompts, suffix, prompt_strengths, masks, mask_strengths, mask_stds)

    def update(
        self,
        background: Optional[Image.Image] = None,
        background_prompt: Optional[str] = None,
        background_negative_prompt: Optional[str] = None,
        idx: Optional[int] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        suffix: Optional[str] = None,
        prompt_strength: Optional[float] = None,
        mask: Optional[Union[torch.Tensor, Image.Image]] = None,
        mask_strength: Optional[float] = None,
        mask_std: Optional[float] = None,
    ) -> None:
        # For lazy update (to solve minor synchonization problem with gradio).
        bq = BackgroundObject(
            image=background,
            prompt=background_prompt,
            negative_prompt=background_negative_prompt,
        )
        if not bq.is_empty:
            self.update_buffer['background'] = bq

        lq = LayerObject(
            idx=idx,
            prompt=prompt,
            negative_prompt=negative_prompt,
            suffix=suffix,
            prompt_strength=prompt_strength,
            mask=mask,
            mask_strength=mask_strength,
            mask_std=mask_std,
        )
        if not lq.is_empty:
            limit = self.update_buffer['layers'].maxlen

            # Optimize the prompt queue: Overrride uncommitted layers with the same idx.
            new_q = deque(maxlen=limit)
            for _ in range(len(self.update_buffer['layers'])):
                # Check from the newest to the oldest.
                # Copy old requests only if the current query does not carry those requests.
                query = self.update_buffer['layers'].pop()
                overriden = lq.merge(query)
                if not overriden:
                    new_q.appendleft(query)
            self.update_buffer['layers'] = new_q

            if len(self.update_buffer['layers']) == limit:
                print(f'[WARNING]  Maximum prompt change query limit ({limit}) is reached. '
                      f'Current query {lq} will be ignored.')
            else:
                self.update_buffer['layers'].append(lq)

    @torch.no_grad()
    def commit(self) -> None:
        flag_changed = self.is_dirty
        bq = self.update_buffer['background']
        lq = self.update_buffer['layers']
        count_bq_req = int(bq is not None and not bq.is_empty)
        count_lq_req = len(lq)

        if flag_changed:
            print(f'[INFO]     Requests found: {count_bq_req} background requests '
                  f'& {count_lq_req} layer requests:\n{str(bq)}, {", ".join([str(l) for l in lq])}')

        bq = self.update_buffer['background']
        if bq is not None:
            self.update_background(**vars(bq))
            self.update_buffer['background'] = None

        while len(lq) > 0:
            l = lq.popleft()
            self.update_single_layer(**vars(l))

        if flag_changed:
            print(f'[INFO]     Requests resolved: {count_bq_req} background requests '
                  f'& {count_lq_req} layer requests.')

    def scheduler_step_batch_flashflowmatching(
        self,
        model_pred_batch: torch.Tensor,
        x_t_latent_batch: torch.Tensor,
        idx: Optional[int] = None,
    ) -> torch.Tensor:
        r"""Denoise-only step for reverse diffusion scheduler.

        Args:
            model_pred_batch (torch.Tensor): Noise prediction results.
            x_t_latent_batch (torch.Tensor): Noisy latent.
            idx (Optional[int]): Instead of timesteps (in [0, 1000]-scale) use
                indices for the timesteps tensor (ranged in
                [0, len(timesteps)-1]). Specify only if a single-index, not
                stream-batched inference is what you want.

        Returns:
            A denoised tensor with the same size as latent.
        """
        x_t_latent_batch = x_t_latent_batch.to(torch.float32)
        if idx is None:
            prev_sample = x_t_latent_batch - model_pred_batch * self.sigmas_[:, None, None, None]
        else:
            prev_sample = x_t_latent_batch - model_pred_batch * self.sigmas[idx]
        return prev_sample.to(self.dtype)

    def scheduler_step_batch(self, *args, **kwargs):
        return self.scheduler_step_batch_flashflowmatching(*args, **kwargs)

    def scheduler_add_noise_flashflowmatching(
        self,
        latent: torch.Tensor,
        noise: Optional[torch.Tensor],
        idx: Optional[Tuple[int, torch.Tensor]] = None,
        s_noise: float = 1.0,
        initial: bool = False,
        batch: bool = False,
    ) -> torch.Tensor:
        r"""Separated noise-add step for the reverse diffusion scheduler.
        
        Designed to match the interface of the original
        `pipe.scheduler.add_noise`.

        Args:
            latent (torch.Tensor): Denoised latent.
            noise (torch.Tensor): Added noise. Can be None. If None, a random
                noise is newly sampled for addition.
            idx (int): Instead of timesteps (in [0, 1000]-scale) use indices
                for the timesteps tensor (ranged in [0, len(timesteps)-1]).

        Returns:
            A noisy tensor with the same size as latent.
        """
        sigmas = self.sigmas[:, None, None, None]
        if idx is not None:
            sigmas = sigmas[idx]
        noise = torch.randn_like(latent) if noise is None else noise
        return (1.0 - sigmas) * latent + sigmas * noise

    def scheduler_add_noise(self, *args, **kwargs):
        return self.scheduler_add_noise_flashflowmatching(*args, **kwargs)

    def transformer_step(
        self,
        x_t_latent: torch.Tensor,  # (T, 16, h, w)
        idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.num_layers
        x_t_latent = x_t_latent.repeat_interleave(p, dim=0)  # (T * p, 16, h, w)

        if self.bootstrap_steps[0] > 0:
            # Background bootstrapping.
            bootstrap_latent = self.scheduler_add_noise(
                self.bootstrap_latent,
                self.stock_noise,
                initial=True,
            )
            x_t_latent = rearrange(x_t_latent, '(t p) c h w -> p t c h w', p=p)
            bootstrap_mask = (
                self.masks * self.bootstrap_steps[None, :, None, None, None]
                + (1.0 - self.bootstrap_steps[None, :, None, None, None])
            ) # (p, t, c, h, w)
            x_t_latent = (1.0 - bootstrap_mask) * bootstrap_latent[None] + bootstrap_mask * x_t_latent
            x_t_latent = rearrange(x_t_latent, 'p t c h w -> (t p) c h w')

            # Centering.
            x_t_latent = shift_to_mask_bbox_center(x_t_latent, rearrange(self.masks, 'p t c h w -> (t p) c h w'), reverse=True)

        t_list = self.sub_timesteps_tensor_  # (T * p,)
        if self.guidance_scale > 1.0 and self.cfg_type == 'initialize':
            x_t_latent_plus_uc = torch.concat([x_t_latent[:p], x_t_latent], dim=0)  # (T * p + 1, 16, h, w)
            t_list = torch.concat([t_list[:p], t_list], dim=0)  # (T * p + 1, 16, h, w)
        elif self.guidance_scale > 1.0 and self.cfg_type == 'full':
            x_t_latent_plus_uc = torch.concat([x_t_latent, x_t_latent], dim=0)  # (2 * T * p, 16, h, w)
            t_list = torch.concat([t_list, t_list], dim=0)  # (2 * T * p,)
        else:
            x_t_latent_plus_uc = x_t_latent  # (T * p, 16, h, w)

        model_pred = self.transformer(
            hidden_states=x_t_latent_plus_uc.to(dtype=self.dtype),  # (B, 16, h, w)
            timestep=t_list,  # (B,)
            encoder_hidden_states=self.prompt_embeds,  # (B, 77, 768)
            pooled_projections=self.pooled_prompt_embeds,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False,
        )[0]  # (B, 16, h, w)

        if self.bootstrap_steps[0] > 0:
            # Uncentering.
            bootstrap_mask = rearrange(self.masks, 'p t c h w -> (t p) c h w')
            if self.guidance_scale > 1.0 and self.cfg_type == 'initialize':
                bootstrap_mask_ = torch.concat([bootstrap_mask[:p], bootstrap_mask], dim=0)
            elif self.guidance_scale > 1.0 and self.cfg_type == 'full':
                bootstrap_mask_ = torch.concat([bootstrap_mask, bootstrap_mask], dim=0)
            else:
                bootstrap_mask_ = bootstrap_mask
            model_pred = shift_to_mask_bbox_center(model_pred, bootstrap_mask_)
            x_t_latent = shift_to_mask_bbox_center(x_t_latent, bootstrap_mask)

            # # Remove leakage (optional).
            # leak = (latent_ - bg_latent_).pow(2).mean(dim=1, keepdim=True)
            # leak_sigmoid = torch.sigmoid(leak / self.bootstrap_leak_sensitivity) * 2 - 1
            # fg_mask_ = fg_mask_ * leak_sigmoid

        ### noise_pred_text, noise_pred_uncond: (T * p, 16, h, w)
        ### self.stock_noise, init_noise: (T, 16, h, w)

        if self.guidance_scale > 1.0 and self.cfg_type == 'initialize':
            noise_pred_text = model_pred[p:]
            self.stock_noise_ = torch.concat([model_pred[:p], self.stock_noise_[p:]], dim=0)
        elif self.guidance_scale > 1.0 and self.cfg_type == 'full':
            noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
        else:
            noise_pred_text = model_pred
        if self.guidance_scale > 1.0 and self.cfg_type in ('self', 'initialize'):
            noise_pred_uncond = self.stock_noise_ * self.delta

        if self.guidance_scale > 1.0 and self.cfg_type != 'none':
            model_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        else:
            model_pred = noise_pred_text

        if self.guidance_scale > 1.0 and self.guidance_rescale > 0.0:
            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_cond, guidance_rescale=guidance_rescale)

        # compute the previous noisy sample x_t -> x_t-1
        denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)

        if self.cfg_type in ('self' , 'initialize'):
            scaled_noise = self.beta_prod_t_sqrt_ * self.stock_noise_
            delta_x = self.scheduler_step_batch(model_pred, scaled_noise, idx)

            # Do mask edit.
            alpha_next = torch.concat([self.alpha_prod_t_sqrt_[p:], torch.ones_like(self.alpha_prod_t_sqrt_[:p])], dim=0)
            delta_x = alpha_next * delta_x
            beta_next = torch.concat([self.beta_prod_t_sqrt_[p:], torch.ones_like(self.beta_prod_t_sqrt_[:p])], dim=0)
            delta_x = delta_x / beta_next
            init_noise = torch.concat([self.init_noise_[p:], self.init_noise_[:p]], dim=0)
            self.stock_noise_ = init_noise + delta_x

        background = self.background.latent.repeat(len(self.t_list), 1, 1, 1)
        background = self.scheduler_add_noise(
            background,
            torch.concat([self.stock_noise[1:], torch.zeros_like(self.stock_noise[:1])], dim=0),
            torch.arange(len(self.t_list), device=self.device).roll(-1),
            initial=True,
        )

        denoised_batch = rearrange(denoised_batch, '(t p) c h w -> p t c h w', p=p)
        latent = (self.masks * denoised_batch).sum(dim=0)  # (T, 16, h, w)
        latent = torch.where(self.counts > 0, latent / self.counts, latent)
        latent = (1 - self.bg_mask) * latent + self.bg_mask * background

        return latent

    @torch.no_grad()
    def __call__(
        self,
        no_decode: bool = False,
        ignore_check_ready: bool = False,
    ) -> Optional[Union[torch.Tensor, Image.Image]]:
        if not ignore_check_ready and not self.check_ready():
            return
        if not ignore_check_ready and self.is_dirty:
            print("I'm dirty!")
            self.commit()
            self.flush()

        if self.seedfix:
            self.reset_seed(self.generator, self.seed)
            latent = self.init_noise[:1]
        else:
            latent = torch.randn((1, self.transformer.config.in_channels, self.latent_height, self.latent_width),
                dtype=self.dtype, device=self.device)  # (1, 16, h, w)
        latent = torch.cat((latent, self.x_t_latent_buffer), dim=0)  # (t, 16, h, w)
        self.stock_noise = torch.cat((self.init_noise[:1], self.stock_noise[:-1]), dim=0)  # (t, 16, h, w)
        if self.cfg_type in ('self', 'initialize'):
            self.stock_noise_ = self.stock_noise.repeat_interleave(self.num_layers, dim=0)  # (T * p, 77, 768)

        x_0_pred_batch = self.transformer_step(latent)

        latent = x_0_pred_batch[-1:]
        self.x_t_latent_buffer = self.scheduler_add_noise(
            x_0_pred_batch[:-1],
            self.init_noise[1:],
            torch.arange(1, len(self.t_list), device=self.device),
            initial=False,
        )

        # For pipeline flushing.
        if no_decode:
            return latent

        imgs = self.decode_latents(latent.half())  # (1, 3, H, W)
        img = T.ToPILImage()(imgs[0].cpu())
        return img

    def flush(self) -> None:
        for _ in self.t_list:
            self(True, True)
        self.ready_checklist['flushed'] = True
