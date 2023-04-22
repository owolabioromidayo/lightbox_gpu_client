import os, base64,  requests
from PIL import Image

def test_realesr():
    image_path = "depag.jpg"
    # The URL to make the GET request to
    url = "http://localhost:3001/github/xinntao/Real-ESRGAN"

    with open(image_path, "rb") as f:
        base64_data = base64.b64encode(f.read()).decode("utf-8")
        json_data = {
            "toggle_anime": False,
            "toggle_face_enhance": False,
            "toggle_4x": True,
            "image_encoded" : base64_data
        }
        
        response = requests.get(url, json=json_data)
        print(response.__dict__)

        # comamnd_str = f'curl -X GET "http://localhost:4000/realesrgan" -H "Content-Type:application/json" -d " {{\\"image_encoded\\": \\"{base64_data}\\", \\"toggle_anime\\": false, \\"toggle_4x\\": true }}" '

        # proc = subprocess.run(["curl", "http://localhost:4000/realesrgan", '-H "Content-Type:application/json"', f'-d " {{\\"image_encoded\\": \\"{base64_data}\\", \\"toggle_anime\\": false, \\"toggle_4x\\": true }}" ']
        #         ,stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # # print(comamnd_str)
        # # os.system(comamnd_str)

        # print(proc.stdout.decode())
        # print(proc.stderr.decode())

def test_sd():
    json_data = {
        "prompt": "something",
        "iters" : 5
    }

    response = requests.post("http://localhost:3001/CompVis/stable-diffusion-v1.4", json=json_data)
    print(response.__dict__)


def test_sd_inpainting():
    im = None
    mask_im = None
    with open("someimg.png", "rb") as f:
        im = base64.b64encode(f.read()).decode("utf-8")

    with open("someimgmask.png", "rb") as f:
        mask_im = base64.b64encode(f.read()).decode("utf-8")

        
    json_data = {
        "prompt": "red haired girl",
        "iters" : 5,
        "image_encoded": im,
        "image_mask_encoded": mask_im
    }

    response = requests.post("http://localhost:3001/runwayml/stable-diffusion-inpainting", json=json_data)
    print(response.__dict__)

def test_sd_outpainting():
    im = None
    mask_im = None
    with open("section_2__512.jpg", "rb") as f:
        im = base64.b64encode(f.read()).decode("utf-8")

    with open("section_2__512_mask.jpg", "rb") as f:
        mask_im = base64.b64encode(f.read()).decode("utf-8")

        
    json_data = {
        "prompt": "red haired girl, house in background, blue skies",
        "iters" : 5,
        "image_encoded": im,
        "image_mask_encoded": mask_im
    }

    response = requests.post("http://localhost:3001/runwayml/stable-diffusion-inpainting/outpainting_addon", json=json_data)
    print(response.__dict__)

def test_sd_img2img():
    im = None
    with open("someimg.png", "rb") as f:
        im = base64.b64encode(f.read()).decode("utf-8")
        
    json_data = {
        "prompt": "red haired girl",
        "iters" : 5,
        "image_encoded": im,
    }

    response = requests.post("http://localhost:3001/CompVis/stable-diffusion-v1.4/img2img_addon", json=json_data)
    print(response.__dict__)

# test_realesr()
# test_sd()
# test_sd_inpainting()
test_sd_img2img()
test_sd_outpainting()