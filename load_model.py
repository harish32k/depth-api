print("Loading model...")
import torch
from img_convert import get_image_fromb64, get_b64_fromimage

model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
midas = torch.hub.load("static/intel-isl_MiDaS_master", path = 'static/dpt_large-midas-2f21e586.pt', \
model=model_type, source ='local')


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("static/intel-isl_MiDaS_master", "transforms", source ='local')
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


#########################################################################################

#count = 0

def pre_process(enc_str):
    img = get_image_fromb64(enc_str)
    input_batch = transform(img).to(device)
    return input_batch, img

def infer(input_batch, img):
    with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
            ).squeeze()

    output = prediction.cpu().numpy()
    return output

def post_process(output):
    #global count
    jpg_as_text = get_b64_fromimage(output)
    #plt.imsave(os.path.join('outs', str(count)+'_old.png') , output)
    #plt.imsave(os.path.join('outs', str(count)+'.png'), get_image_fromb64(jpg_as_text))
    #count += 1
    return jpg_as_text

print("Model loaded.")