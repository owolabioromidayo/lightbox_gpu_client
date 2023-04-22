import os, sys, json, random, base64, copy,  gc
from PIL import Image
import numpy as np

import torch
from torch import autocast
# from transformers import CLIPImageProcesssor
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionInpaintPipeline, 
    StableDiffusionImg2ImgPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionInstructPix2PixPipeline, 
    EulerAncestralDiscreteScheduler
)

# os.environ["CURL_CA_BUNDLE"]=""


class ModelBroadcast:
    def __init__(self, api_token, test=False):
        self.test  = test
        self.api_token = api_token
        self.broadcast = {
            "api_id": api_token, 
            "models": {
                'CompVis/stable-diffusion-v1.4' : {
                    "input": {
                        "prompt": "<lstr>",
                        "iters": "<int>"
                    },
                     "func": self.run_sd if not self.test else self.__test_placeholder,
                    "output": {
                        "images_encoded": "<List<b64>>"
                    },
                    "desc": """
                        This is the CompVis/stable-diffusion-1.4 model
                        prompt <lstr> -> The prompt of what you are trying to generate
                        iters <int> -> The number of images you wish to generate 
                        """
                }, 
                'wavymulder/Analog-Diffusion' : {
                    "input": {
                        "prompt": "<lstr>",
                        "iters": "<int>"
                    },
                "func": self.run_analog_diffusion if not self.test else self.__test_placeholder,
                    "output": {
                        "images_encoded": "<List<b64>>"
                    },
                    "desc": """
                        Analog DIffusion by Wavy Mulder \n \
                        prompt <lstr> -> The prompt of what you are trying to generate
                        iters <int> -> The number of images you wish to generate 
                        """
                }, 
                'runwayml/stable-diffusion-inpainting' : {
                    "input": {
                        "prompt": "<lstr>",
                        "iters": "<int>",
                        "image_encoded": "<b64>",
                        "image_mask_encoded": "mask<b64>"
                    },
                    "func": self.run_sd_inpainting if not self.test else self.__test_placeholder,
                    "output": {
                        "images_encoded": "list<<b64>>"
                    },
                    "desc": """
                        this is the runwayml/stable-diffusion-inpainting model
                        prompt <lstr> -> the prompt of what you are trying to generate
                        iters <int> -> the number of images you wish to generate 
                        "image_encoded": "<b64>"
                        "image_mask_encoded": "mask<b64>"
                        """
                },
                'runwayml/stable-diffusion-inpainting/outpainting_addon' : {
                    "input": {
                        "prompt": "<lstr>",
                        "iters": "<int>",
                        "image_encoded": "<b64>",
                    },
                    "func": self.run_sd_outpainting if not self.test else self.__test_placeholder,
                    "output": {
                        "images_encoded": "list<<b64>>"
                    },
                    "desc": """
                        this is the runwayml/stable-diffusion-inpainting/outpainting_addon model. Given a
                        512x512 image, it sections and outpaints the image into a 1024x1024 image based on the prompt
                        prompt <lstr> -> the prompt of what you are trying to generate
                        iters <int> -> the number of images you wish to generate 
                        "image_encoded": "<b64>"
                        """
                },
                'CompVis/stable-diffusion-v1.4/img2img_addon' : {
                    "input": {
                        "prompt": "<lstr>",
                        "iters": "<int>",
                        "image_encoded": "<b64>",
                    },
                    "func": self.run_sd_img2img if not self.test else self.__test_placeholder,
                    "output": {
                        "images_encoded": "list<<b64>>"
                    },
                    "desc": """
                        this is the compvis/stable-diffusion-1.4/img2img_adddon model
                        "prompt": "<lstr>",
                        "iters": "<int>",
                        "image_encoded": "<b64>",
                        """
                },
                'timbroooks/instruct-pix2pix' : {
                    "input": {
                        "prompt": "<lstr>",
                        "iters": "<int>",
                        "image_encoded": "<b64>",
                    },
                    "func": self.run_instructpix2pix if not self.test else self.__test_placeholder,
                    "output": {
                        "images_encoded": "list<<b64>>"
                    },
                    "desc": """
                        this is the timbrooks/instruct-pix2pix model. It performs better than img2img in
                        understanding prompts and their context in the aimge. <paper link>
                        "prompt": "<lstr>",
                        "iters": "<int>",
                        "image_encoded": "<b64>",
                        """
                },
                'github/xinntao/Real-ESRGAN' : {
                    "input": {
                        "image_encoded": "<b64>",
                        "toggle_face_enhance": "<bool>",
                    },
                    "func": self.run_realesrgan if not self.test else self.__test_placeholder,
                    "output": {
                        "image_encoded": "list<<b64>>"
                    },
                    "desc": """
                        This is the  github/xinntao/Real-ESRGAN model
                        "image_encoded": "<b64>",
                        "toggle_face_enhance": "<bool>",
                        """
                },
                'github/danielgatis/rembg' : {
                    "input": {
                        "image_encoded": "<b64>",
                    },
                    "func": self.run_rembg if not self.test else self.__test_placeholder,
                    "output": {
                        "image_encoded": "list<<b64>>"
                    },
                    "desc": """
                        This is the  danielgatis/rembg model. It removes the background from images. Takes 
                        one image, returns one.
                        "image_encoded": "<b64>",
                        """
                }
            }
        }





    def __test_placeholder(self, _json):
        #just return an encoded image to not stress system during production
        # with open("format", "w") as f:
        #     f.write(f"{_json}/n ")
        
       
        image_encoded= _json["image_encoded"]
        iters = None
        try:
            iters = _json["iters"]
        except:
            iters = 3

        images = []
        with open(f"placeholder.jpg", "rb") as f:
                data = f.read()
                base64_data = "data:image/jpeg;base64," + base64.b64encode(data).decode("utf-8")
                images.append(base64_data)
            # images.append("data:image/jpeg;base64," + image_encoded)
        print("Loaded images")
        return images*3
    

    def _get_img_section(self, xs,xe,ys,ye,img,sdim):

        im = np.zeros((sdim, sdim,3))
        im_mask = np.zeros((sdim, sdim,3))
        im_mask[:, :] = (255,255,255)

        if (xs,xe,ys,ye) == (0, 256, 0, 256):
            im[256: , 256: ] = img[xs:xe, ys:ye]
            im_mask[256:, 256: ] = (0,0,0)

        if (xs,xe,ys,ye) == (0, 256, 256,512):
            im[256: , :256 ] = img[xs:xe, ys:ye]
            im_mask[256:, :256 ] = (0,0,0)

        if (xs,xe,ys,ye) == (256, 512, 0, 256):
            im[:256 , 256: ] = img[xs:xe, ys:ye]
            im_mask[:256, 256: ] = (0,0,0)
            
        if (xs,xe,ys,ye) == (256, 512, 256,512):
            im[:256 , :256 ] = img[xs:xe, ys:ye]
            im_mask[:256, :256 ] = (0,0,0)

        return im, im_mask

    def _enlarge_512(self, pipe, prompt, image_fname):

        im = Image.open(image_fname)
        im_arr = np.array(im)
        im_arr = im_arr[:, :, :3]


        def run_inpaint(im, im_mask, prompt):
            with autocast("cuda"):
                #The mask structure is white for inpainting and black for keeping as is
                image = pipe(prompt=prompt, image=im, mask_image=im_mask).images[0]
                return np.array(image)

        final_im = np.zeros((1024,1024,3))

        sections = [
            [0,256,0,256],
            [0,256,256,512],
            [256,512,0,256],
            [256,512,256,512],
        ]

        for xs,xe,ys,ye in sections:
            im, im_mask = self._get_img_section(xs, xe, ys, ye, im_arr, 512)
            large_im = run_inpaint(Image.fromarray(im.astype("uint8")), 
                        Image.fromarray(im_mask.astype("uint8")), prompt)

            final_im[xs*2: xe*2, ys*2:ye*2] = large_im[:,:]

        return final_im

 

    def get_desc(self):
        desc = copy.deepcopy(self.broadcast)
        for k in desc["models"].keys():
            del desc["models"][k]["func"]

        return desc

    def init_sd(self):
        model_id = "C:\\Users\\Admin\\.cache\\huggingface\\diffusers\\models--CompVis--stable-diffusion-v1-4\\snapshots\\2880f2ca379f41b0226444936bb7a6766a227587"
        device = "cuda"

        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            revision="fp16", 
        ).to(device)

        pipe.enable_attention_slicing()
        return pipe

    def init_sd_analog(self):
        model_id = "C:\\Users\\Admin\\.cache\\huggingface\\diffusers\\models--wavymulder--Analog-Diffusion\\snapshots\\f8dd6d9fab77a226582695c101eab04841e3cd4b"
        device = "cuda"

        scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")

        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            scheduler=scheduler,
            # revision="fp16", 
        )
        pipe = pipe.to(device)
        pipe.enable_attention_slicing()

        return pipe

    def init_sd_inpainting(self):
        device = "cuda"
        model_id = "C:\\Users\\Admin\\.cache\\huggingface\\diffusers\\models--runwayml--stable-diffusion-inpainting\\snapshots\\afeee10def38be19995784bcc811882409d066e5"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id, 
            revision="fp16",
            torch_dtype=torch.float16,
        ).to(device)

        pipe.enable_attention_slicing()
        return pipe
    
    def init_sd_img2img(self):
        model_id = "C:\\Users\\Admin\\.cache\\huggingface\\diffusers\\models--CompVis--stable-diffusion-v1-4\\snapshots\\2880f2ca379f41b0226444936bb7a6766a227587"
        device = "cuda"

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            revision="fp16", 
        ).to(device)

        pipe.enable_attention_slicing()
        return pipe

    def init_instructpix2pix(self):
        model_id = "C:\\Users\\Admin\\.cache\\huggingface\\diffusers\\\\models--timbrooks--instruct-pix2pix\\snapshots\\93224554bd65f19b6f0c99cbcce3a4ac59bb6382"
        device = "cuda"

        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, 
            torch_dtype=torch.float16, safety_checker=None)
        pipe.to(device)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

        pipe.enable_attention_slicing()
        return pipe

    def init_sd_outpainting(self):
        return self.init_sd_inpainting()

    def run_sd(self,_json):
        pipe = self.init_sd()

        prompt = _json["prompt"]
        iters = _json["iters"]

        folder_name = prompt.replace(" ", "_")
        _dir = f'{random.randint(0,1000000)}_{folder_name}_sd'
        os.mkdir(_dir)

        with autocast("cuda"):
            print("in cuda")
            for i in range(int(iters)):
                seed = random.randrange(1000000000)
                generator = torch.Generator("cuda").manual_seed(seed)
                images = pipe(
                    prompt, 
                    generator=generator,
                    guidance_scale=7.5
                    )

                print("image generated")

                images["images"][0].save(f"{_dir}\\{i}_{seed}.jpg")   


        pipe = None
        gc.collect()
        torch.cuda.empty_cache()

        images_encoded  = self._get_encoded_images_from_dir(_dir)
        os.rmdir(_dir)
        return images_encoded


    def run_analog_diffusion(self,_json):
        pipe = self.init_sd_analog()

        prompt = _json["prompt"]
        iters = _json["iters"]

        folder_name = prompt.replace(" ", "_")
        _dir = f'{random.randint(0,1000000)}_analog_{folder_name}_sd'
        os.mkdir(_dir)

        with autocast("cuda"):
            print("in cuda")
            for i in range(int(iters)):
                seed = random.randrange(1000000000)
                generator = torch.Generator("cuda").manual_seed(seed)
                images = pipe(
                    prompt, 
                    generator=generator,
                    guidance_scale=7.5
                    )

                print("image generated")

                images["images"][0].save(f"{_dir}\\{i}_{seed}.jpg")   

        pipe = None
        gc.collect()
        torch.cuda.empty_cache()

        images_encoded  = self._get_encoded_images_from_dir(_dir)
        os.rmdir(_dir)
        return images_encoded


    def run_sd_inpainting(self, _json):
        pipe = self.init_sd_inpainting()

        prompt = _json["prompt"]
        iters = _json["iters"]
        image_encoded = _json["image_encoded"]
        image_mask_encoded = _json["image_mask_encoded"]
        
        folder_name = prompt.replace(" ", "_")
        _dir = f'{random.randint(0,1000000)}_{folder_name}_sd_inpainting'
        os.mkdir(_dir)

        image_bin = base64.b64decode(image_encoded)
        image_mask_bin = base64.b64decode(image_mask_encoded)

        #save images to files, then read them in from PIL.Image.open
        with open(f"{_dir}/image.png", "wb") as f:
            f.write(image_bin)

        with open(f"{_dir}/image_mask.png", "wb") as f:
            f.write(image_mask_bin)
            
        image = Image.open(f"{_dir}/image.png")

        mask_image = Image.open(f"{_dir}/image_mask.png")
        mask_image = mask_image.resize((512,512))

        with autocast("cuda"):
            for i in range(int(iters)):
                seed = random.randrange(1000000000)
                generator = torch.Generator("cuda").manual_seed(seed)
                
                image = pipe(prompt=prompt, 
                image=image, 
                mask_image=mask_image, 
                generator=generator, 
                guidance_scale=7.5
                ).images[0]

                image.save(f"{_dir}\\{i}_{seed}.jpg")

        pipe = None
        gc.collect()
        torch.cuda.empty_cache()

        images_encoded  = self._get_encoded_images_from_dir(_dir)
        os.rmdir(_dir)
        return images_encoded 


    def run_sd_img2img(self, _json):
        pipe = self.init_sd_img2img()

        prompt = _json["prompt"]
        iters = _json["iters"]
        image_encoded = _json["image_encoded"]
        
        folder_name = prompt.replace(" ", "_")
        _dir = f'{random.randint(0,1000000)}_{folder_name}_sdimg2img'
        os.mkdir(_dir)

        #save images to files, then read them in from PIL.Image.open
        image_bin = base64.b64decode(image_encoded)

        with open(f"{_dir}/image.jpg", "wb") as f:
            f.write(image_bin)
            
        image = Image.open(f"{_dir}/image.jpg")

        with autocast("cuda"):
            for i in range(int(iters)):
                seed = random.randrange(1000000000)
                generator = torch.Generator("cuda").manual_seed(seed)
                
                image = pipe(prompt=prompt, 
                generator=generator, init_image=image, strength=0.7 ).images[0]

                image.save(f"{_dir}\\{i}_{seed}.jpg")

        pipe = None
        gc.collect()
        torch.cuda.empty_cache()

        images_encoded  = self._get_encoded_images_from_dir(_dir)
        os.rmdir(_dir)
        return images_encoded 



    def run_instructpix2pix(self, _json):
        pipe = self.init_instructpix2pix()

        prompt = _json["prompt"]
        iters = _json["iters"]
        image_encoded = _json["image_encoded"]
        
        folder_name = prompt.replace(" ", "_")
        _dir = f'{random.randint(0,1000000)}_{folder_name}_pix2pix'
        os.mkdir(_dir)

        #save images to files, then read them in from PIL.Image.open
        image_bin = base64.b64decode(image_encoded)

        with open(f"{_dir}/image.jpg", "wb") as f:
            f.write(image_bin)
            
        image = Image.open(f"{_dir}/image.jpg").convert("RGB")

        with autocast("cuda"):
            for i in range(int(iters)):
                seed = random.randrange(1000000000)
                generator = torch.Generator("cuda").manual_seed(seed)
                
                image = pipe(prompt=prompt, image=image, generator=generator, num_inference_steps=30, image_guidance_scale=1 ).images[0]

                image.save(f"{_dir}\\{i}_{seed}.jpg")

        pipe = None
        gc.collect()
        torch.cuda.empty_cache()

        images_encoded  = self._get_encoded_images_from_dir(_dir)
        os.rmdir(_dir)
        return images_encoded 

    def run_sd_outpainting(self, _json):
        pipe = self.init_sd_outpainting()

        prompt = _json["prompt"]
        iters = int(_json["iters"])
        image_encoded = _json["image_encoded"]
        res = []
         
        folder_name = prompt.replace(" ", "_")
        _dir = f'{random.randint(0,1000000)}_{folder_name}_sd_outpainting'
        os.mkdir(_dir)

        image_bin = base64.b64decode(image_encoded)

        #save images to files, then read them in from PIL.Image.open
        with open(f"{_dir}/image.png", "wb") as f:
            f.write(image_bin)
 
        for i in range(iters):
            enlarged_image = self._enlarge_512(pipe, prompt, f"{_dir}\\image.png")
            Image.fromarray(enlarged_image.astype("uint8")).save(f"{_dir}\\big_outpainting_{i}.jpg")

        # os.remove(f"{_dir}\\image.png")


        pipe = None
        gc.collect()
        torch.cuda.empty_cache()

        images_encoded  = self._get_encoded_images_from_dir(_dir)
        os.rmdir(_dir)
        return images_encoded 


    def run_realesrgan(self, _json):
        realesrgan_path = "C:\\Users\\Admin\\Desktop\\SD\\Real-ESRGAN"
        # realesrgan_exec_path = f"{realesrgan_path}\\inference_realesrgan.py"
        out_path = " C:\\Users\\Admin\\Desktop\\Lightbox\\be\\main_be"

        image_encoded= _json["image_encoded"]
        toggle_face_enhance= _json["toggle_face_enhance"]

        #temp switch to esrgan folder
        curr_dir = os.getcwd()
        print(curr_dir)
        os.chdir(realesrgan_path)
        im_name = f"{random.randint(0,100000)}_esrgan"
        input_image_path = f"{im_name}.jpg"
        output_image_path = f"{im_name}_out.jpg"

        image_bin = base64.b64decode(image_encoded)
        with open(input_image_path, "wb") as f:
            f.write(image_bin)

        if toggle_face_enhance:
            os.system(f"python inference_realesrgan.py -n RealESRGAN_x4plus -i {input_image_path} -o {out_path} --face_enhance")
        else:
            os.system(f"python inference_realesrgan.py -n RealESRGAN_x4plus -i {input_image_path} -o {out_path}")

        
        os.chdir(curr_dir)
        transformed_img = None
        with open(output_image_path, "rb") as f:
                    data = f.read()
                    transformed_img = "data:image/png;base64," + base64.b64encode(data).decode("utf-8")

                    print(transformed_img)

        os.remove(input_image_path)
        os.remove(output_image_path)

        return [transformed_img] 
    


    def run_rembg(self, _json):
        image_encoded= _json["image_encoded"]

        im_name = f"{random.randint(0,100000)}_rembg"
        input_image_path = f"{im_name}.jpg"
        output_image_path = f"{im_name}_out.jpg"

        image_bin = base64.b64decode(image_encoded)
        with open(input_image_path, "wb") as f:
            f.write(image_bin)

        os.system(f"rembg i -m u2net {input_image_path} {output_image_path}")

        transformed_img = None
        with open(output_image_path, "rb") as f:
                    data = f.read()
                    transformed_img = "data:image/jpeg;base64," + base64.b64encode(data).decode("utf-8")

        os.remove(output_image_path)
        return [transformed_img] 
    



    def _get_encoded_images_from_dir(self, _dir): 
        images_encoded = []

        for _, _, filelist in os.walk(_dir):
            for fname in filelist:
                if fname.endswith(".jpg"):
                    filepath = os.path.join(_dir, fname)
                    with open(filepath, "rb") as f:
                        data = f.read()
                        base64_data = "data:image/jpeg;base64," + base64.b64encode(data).decode("utf-8")
                        images_encoded.append(base64_data)

        return images_encoded

        #clean up (delete folder)