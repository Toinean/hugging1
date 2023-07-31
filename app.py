from flask import Flask, render_template, request
from PIL import Image
from io import BytesIO
from transformers import ViTImageProcessor, AutoTokenizer

import pickle
import os
import urllib.request


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(APP_ROOT, 'finalized_model.pkl')


model = pickle.load(open('finalized_model.pkl', 'rb'))
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route("/forward/", methods=['POST'])
def move_forward():
    #Moving forward code
    url_link = request.form.get('fname')
    image_input = get_image(url_link)
    caption = predict_step(image_input,model)
    print(url_link)
    return render_template('index.html', caption_message=caption, caption_img=url_link)

def get_image(url):
    if 'https' not in url:
        url = 'https:' + url
    req = urllib.request.Request(url,
                                 headers={
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
    })
    response = urllib.request.urlopen(req)
    content = response.read()
    img = Image.open(BytesIO(content))
    return img.convert('RGB')

max_length = 30
# num_beams = 4
gen_kwargs = {"max_length": max_length}

def predict_step(image, model):
  pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
  # pixel_values = pixel_values.to(device)
  output_ids = model.generate(pixel_values, **gen_kwargs)
  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds

if __name__ == '__main__': app.run()