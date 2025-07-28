from PIL import Image
import numpy as np
import math
from typing import Tuple, Union, Optional, List, Dict
import PIL

def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280):

    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def resize(image: np.ndarray, size: Tuple[int, int]):

    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")

    image = image.transpose(1, 2, 0)
    image = Image.fromarray(image)
    image = image.resize(size, resample=Image.Resampling.BILINEAR)
    image = np.array(image).transpose(2, 0, 1)
    return image


def normalize(image: np.ndarray, mean: float, std: float):

    if not np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)

    mean = np.array(mean)[:, None, None]
    std = np.array(std)[:, None, None]

    return (image - mean) / std



class HetuImageProcessor():
    def __init__(self, do_resize = True, do_normalize = True, image_mean = [0.48145466, 0.4578275, 0.40821073], 
                image_std = [0.26862954, 0.26130258, 0.27577711], patch_size = 14, min_pixels =  56 * 56, 
                max_pixels = 28 * 28 * 1280, temporal_patch_size = 1):
        self.do_resize = do_resize
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.patch_size = patch_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.temporal_patch_size = temporal_patch_size
        assert(temporal_patch_size == 1)

    def _preprocess(self, images, do_resize, do_normalize, image_mean, image_std):

        if isinstance(images[0], Image.Image):
            images = [image.numpy() for image in images]
        else:
            for image in images:
                assert(isinstance(image, np.ndarray)), "Image must be a NumPy array."

        # 保证每个image都是三通道，并且通道维度在0号维度上
        for idx, image in enumerate(images):
            # Ensure the image has 3 dimensions
            if image.ndim == 2:  # Single-channel grayscale, shape (H, W)
                image = np.stack([image] * 3, axis=0)  # Convert to (C, H, W)
            elif image.ndim == 3:
                if image.shape[0] == 1:  # Shape (1, H, W)
                    image = np.repeat(image, repeats=3, axis=0)  # Expand channel
                elif image.shape[2] == 1:  # Shape (H, W, 1)
                    image = np.repeat(image, repeats=3, axis=-1)  # Expand channel to (H, W, 3)

            # Ensure image is in (C, H, W) format
            if image.ndim == 3 and image.shape[2] in (1, 3):  # (H, W, C) format
                image = image.transpose(2, 0, 1)  # Convert to (C, H, W)

            # Final validation: Image must have 3 channels
            if image.shape[0] != 3:
                raise ValueError(f"Image at index {idx} must have 3 channels. Current shape: {image.shape}")

            # Replace the image in the list
            images[idx] = image
        
        height, width = images[0].shape[1], images[0].shape[2]
        resized_height, resized_width = height, width
        for image in images:
            processed_images = []
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=self.patch_size,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                )
                image = resize(image, size=(resized_height, resized_width))
            
            if do_normalize:
                image = normalize(image=image, mean=image_mean, std=image_std)

            processed_images.append(image)

        
        patches = np.array(processed_images)    
        if patches.shape[0] == 1:  
            patches = np.tile(patches, (self.temporal_patch_size, 1, 1, 1))
        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            3,
            grid_h,
            self.patch_size,
            grid_w,
            self.patch_size,
        )

        patches = patches.transpose(0, 3, 5, 1, 4, 6, 2)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * self.temporal_patch_size * self.patch_size * self.patch_size
        )
        return flatten_patches, (grid_t, grid_h, grid_w)
    

    def preprocess(self, images = None, videos = None, do_resize = True, do_normalize = True,
                    image_mean = [0.48145466, 0.4578275, 0.40821073], image_std = [0.26862954, 0.26130258, 0.27577711]):
        
        image_pixel_values, image_grid_thws = [], []
        if images is not None:
            
            for image in images:
                if not isinstance(image, list):
                    image = [image]
                patches, image_grid_thw = self._preprocess(
                    image,
                    do_resize=do_resize,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                )
                image_pixel_values.extend(patches)
                image_grid_thws.append(image_grid_thw)
            image_grid_thws = np.array(image_grid_thws)
        
        video_pixel_values, video_grid_thws = [], []
        if videos is not None:
            pixel_values, vision_grid_thws = [], []
            for images in videos:
                patches, video_grid_thw = self._preprocess(
                    images,
                    do_resize=do_resize,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                )
                video_pixel_values.extend(patches)
                video_grid_thws.append(video_grid_thw)
            video_grid_thws = np.array(video_grid_thws)
        
        return {"image_pixel_values": image_pixel_values, "image_grid_thws": image_grid_thws,
                "video_pixel_values": video_pixel_values, "video_grid_thws": video_grid_thws}  

