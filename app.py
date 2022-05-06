from flask import Flask, jsonify, request
import load_model as model

app = Flask(__name__)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    #img_names = ['img', 'img1', 'img2', 'img3', 'img4']
    print("received image(s)")
    to_return = {}
    req = request.json
    for img_name in req.keys():
        print("preprocessing")
        input_batch, img = model.pre_process(req.get(img_name))
        print("infering")
        output = model.infer(input_batch, img)
        print("postprocessing")
        req_img = model.post_process(output)
        to_return[img_name] = req_img
    
    return jsonify(to_return)


@app.route('/healthz')
def healthz():
    return "OK"
    
@app.route('/health', methods=['GET'])
def health_check():
   return Response(response=json.dumps({"status": 'healthy'}), status=200, mimetype="application/json")


if __name__=='__main__':
    app.run(host='0.0.0.0', debug=False) # use_reloader=False)
