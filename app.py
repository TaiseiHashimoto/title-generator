from flask import (
    Flask,
    request,
    render_template,
)
import argparse
import numpy as np
import torch
import pickle

from models import arXivModel
import utils


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    abstract = ""
    samples = []

    if request.method == 'POST':
        abstract = request.form['abstract']
        abstract = abstract.strip().replace('\n', ' ')
        abstract = utils.replace_special_tokens(abstract)

        idx = np.random.choice(len(data['title']), size=args.num_samples)
        title = [data['title'][i] for i in idx]
        title_pos = [data['title_pos'][i] for i in idx]

        with torch.no_grad():
            style_code_dist = model.style_encoder({'title_pos': title_pos})
            style_code = style_code_dist.rsample()

        for i in range(args.num_samples):
            generated = model.generate(
                {'abstract': [abstract]},
                style_code=style_code[[i]],
                num_beams=args.num_beams)
            # generated = [abstract[:50]]
            samples.append({
                'generated': generated[0],
                'template': title[i],
            })

    return render_template('index.html', abstract=abstract, samples=samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8790)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--num_beams', type=int, default=4)
    args = parser.parse_args()

    with open('data/preprocessed.pkl', 'rb') as f:
        data = pickle.load(f)

    _, _, data = utils.split_data(data)
    print("data loaded", flush=True)

    model = arXivModel.from_checkpoint('./checkpoint', device='cuda')
    assert model.style_encoder is not None
    # model = None
    print("model loaded", flush=True)

    # app.run(debug=True, port=args.port)
    app.run(port=args.port)
