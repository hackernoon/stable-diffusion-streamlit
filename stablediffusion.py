#!/usr/bin/env python
# coding: utf-8

# # Log in to HuggingFace for access to models
# 
# In order to access the models from CompVis for Stable Diffusion, you must follow three steps:
# 
# 1. You must acknowledge and agree to their user requirements and license for their models. you can do so by reading the instructions found on this page: https://huggingface.co/CompVis/stable-diffusion-v1-4
# 
# 2. You must login to Huggingface, and then create and retrieve an access token (found here: https://huggingface.co/settings/tokens)
# 
# 3. Finally, replace the segment of the cell below `<your_huggingface_token>` with your own token, and run the cell. 
# 
# If you follow these steps, you will be able to access the model for free!
# 
# 

# In[2]:


# get_ipython().system('python login.py --token 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')

def run(promp):
    import os
    from pathlib import Path
    os.mkdir('/root/.huggingface')

    Path('/root/.huggingface/token').touch()
    with open('/root/.huggingface/token', 'w') as f:
        f.write(os.getenv("HUGGING_FACE"))


# In[3]:


# get_ipython().system('mkdir outputs')


# # Inference
# 
# In order to generate an image, you simply need to run one of the two cells below. The first cell is optimized for low power GPUs, like the Free GPU M4000, and will be able to generate an image on any GPU powered Gradient Machine. 
# 
# The next cell, is optimized to run on more powerful GPU setups, like an A100 or A6000, and can be used to quickly generate high quality images on these machines. 

# In[ ]:


# Low cost image generation - works on Free GPU!

    import torch
    from torch import autocast
    from diffusers import StableDiffusionPipeline

    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, revision="fp16", use_auth_token=True)
    pipe = pipe.to(device)

    sample_num = 5
    lst = []
    prompt = "Kermit the Frog playing pinball"
    for i in range(sample_num):
        with autocast("cuda"):
            lst.append(pipe(prompt, guidance_scale=17.5,
                    height=256, width=256)["sample"][0])


    for i in range(sample_num):
        lst[i].save(f'outputs/gen-image-{i}.png')


# In[ ]:


# High cost - FP32

# import torch
# from torch import autocast
# from diffusers import StableDiffusionPipeline

# model_id = "CompVis/stable-diffusion-v1-4"
# device = "cuda"


# pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
# pipe = pipe.to(device)

# sample_num = 10
# lst = []
# prompt = "Kermit the Frog on the Iron Throne"
# for i in range(sample_num):
        # with autocast("cuda"):
            # lst.append(pipe(prompt, guidance_scale=7.5)["sample"][0])

        
# for i in range(sample_num):
        # lst[i].save(f'outputs/gen-image-{i}.png')



# # CLIP reranking
# 
# Find the best of your images from the generated samples by ranking them with CLIP by their accuracy to the prompt. 
# 
# Code adapted from Boris Dayma's [DALL-E Mini Inference Pipeline](https://github.com/borisdayma/dalle-mini/blob/main/tools/inference/inference_pipeline.ipynb) 

# In[ ]:


    from transformers import CLIPProcessor, FlaxCLIPModel
    import jax
    import jax.numpy as jnp
    from flax.jax_utils import replicate
    from functools import partial


# CLIP model
    CLIP_REPO = "openai/clip-vit-base-patch32"
    CLIP_COMMIT_ID = None

# Load CLIP
    clip, clip_params = FlaxCLIPModel.from_pretrained(
        CLIP_REPO, revision=CLIP_COMMIT_ID, dtype=jnp.float16, _do_init=False
    )
    clip_processor = CLIPProcessor.from_pretrained(CLIP_REPO, revision=CLIP_COMMIT_ID)
    clip_params = replicate(clip_params)

# score images
    @partial(jax.pmap, axis_name="batch")
    def p_clip(inputs, params):
        logits = clip(params=params, **inputs).logits_per_image
        return logits


# In[ ]:


    from flax.training.common_utils import shard
    import numpy as np
# get clip scores
    clip_inputs = clip_processor(
        text=prompt * jax.device_count(),
        images=lst,
        return_tensors="np",
        padding="max_length",
        max_length=77,
        truncation=True,
    ).data
    logits = p_clip(shard(clip_inputs), clip_params)

    out = list(reversed(sorted(zip(logits[0], lst))))

    return out


# In[ ]:


# for idx, v in enumerate(out):
        # display(v[1])
        # print(f"Score: {v[0][0]:.2f}\n")
        

