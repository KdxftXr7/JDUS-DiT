from diffusers import PixArtSigmaPipeline

model_id = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"

cache_dir = None

save_path = "./pretrained_models/"  # change accordingly

pipe = PixArtSigmaPipeline.from_pretrained(model_id, cache_dir = cache_dir)
pipe.save_pretrained(save_path)
