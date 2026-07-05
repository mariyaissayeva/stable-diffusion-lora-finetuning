# Stable Diffusion Fine-Tuning on ArtBench

Teaching Stable Diffusion a painterly style by training a small slice of its attention layers.

## What the notebook does

The notebook downloads the ArtBench-10 sample dataset from Hugging Face and saves the first 250 paintings as the training set, resizing them to 512x512 (the originals are 256x256). It then loads Stable Diffusion v1.5 and generates a set of baseline images with cubism-themed prompts, using the DPM-Solver scheduler and attention slicing to keep inference light.

For training, the whole model is frozen except the query and value projection layers inside the UNet's attention blocks, the same modules that LoRA normally targets. Those weights are trained directly for one epoch (250 steps at batch size 1) with AdamW at a learning rate of 1e-6, gradient clipping, and the standard noise-prediction objective in latent space. The trained UNet weights are saved, the same style prompts are generated again, and the before and after images are shown side by side. A short evaluation loop reports the average reconstruction loss over the training set (0.177 in the recorded run).

## Tech

Python, PyTorch, Diffusers, Transformers / CLIP

## Notes

The style shift is modest because training runs for just one epoch on 250 images of mixed styles with a single generic caption. The biggest limitation is that the dataset is not filtered to one art style, so there is no single style for the model to learn. Filtering to one style, writing captions for each image, and training real LoRA adapters through PEFT would sharpen the effect. This workflow can be used for adapting image generators to a studio's house style or a brand's visual look.
