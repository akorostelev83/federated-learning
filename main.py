# https://js.tensorflow.org/api/latest/#io.http
# https://gist.github.com/dsmilkov/1b6046fd6132d7408d5257b0976f7864
# https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html#using-keras-models-with-tensorflow

# pip install -U flask flask-cors tensorflow tensorflowjs

import io
from flask import Flask, Response, request, render_template
from flask_cors import CORS, cross_origin
import tensorflow as tf
import tensorflowjs as tfjs
import werkzeug.formparser
import numpy as np
import tinys3

try:
    import googleclouddebugger
    googleclouddebugger.enable()
except ImportError:
    pass

class ModelReceiver(object):

    def __init__(self):
        self._model = None
        self._model_json_bytes = None
        self._model_json_writer = None
        self._weight_bytes = None
        self._weight_writer = None

    @property
    def model(self):
        self._model_json_writer.flush()
        self._weight_writer.flush()
        self._model_json_writer.seek(0)
        self._weight_writer.seek(0)

        json_content = self._model_json_bytes.read()
        weights_content = self._weight_bytes.read()
        return tfjs.converters.deserialize_keras_model(
            json_content,
            weight_data=[weights_content],
            use_unique_name_scope=True)

    def stream_factory(self,
                        total_content_length,
                        content_type,
                        filename,
                        content_length=None):
        # Note: this example code isnot* thread-safe.
        if filename == 'model.json':
            self._model_json_bytes = io.BytesIO()
            self._model_json_writer = io.BufferedWriter(self._model_json_bytes)
            return self._model_json_writer
        elif filename == 'model.weights.bin':
            self._weight_bytes = io.BytesIO()
            self._weight_writer = io.BufferedWriter(self._weight_bytes)
            return self._weight_writer


app = Flask('model-server')
CORS(app)
app.config['CORS_HEADER'] = 'Content-Type'

model_receiver = ModelReceiver()

@app.route("/")
def main():
    return render_template("tensor-flow-js-model-upload.html")

@app.route('/upload', methods=['GET','POST'])
@cross_origin()
def upload():
    print('Handling request...')
    werkzeug.formparser.parse_form_data(request.environ, stream_factory=model_receiver.stream_factory)
    print('Received model:')
    with tf.Graph().as_default(), tf.Session():
        received_model = model_receiver.model
        received_model_weights = received_model.get_weights()     
            
        federated_model = tf.keras.models.load_model('federated_model.h5')            
        federated_model_weights = federated_model.get_weights()           

        assert len(received_model_weights) == len(federated_model_weights), "the model weights length must be equal"

        averaged_weights=[]
        for i in range(len(received_model_weights)):
            model_averaged_weights = np.mean(np.array([received_model_weights[i],federated_model_weights[i]]), axis=0)
            averaged_weights.append(model_averaged_weights)

        federated_model.set_weights(np.array(averaged_weights))
        tfjs.converters.save_keras_model(federated_model, './federated_model_js')
        federated_model.save('federated_model.h5')
        upload_model_to_s3_bucket()
        print('model saved to s3')
    return Response(status=200)


def upload_model_to_s3_bucket():
    s3_bucket = 'your-s3-bucket-name-goes-here'
    conn = tinys3.Connection('your-s3-client-id-goes-here', 'your-s3-secret-goes-here', tls=True) 
    model_json = open('./federated_model_js/model.json', 'rb')
    model_bin = open('./federated_model_js/group1-shard1of1.bin', 'rb')
    conn.upload('model.json', model_json, s3_bucket)     
    conn.upload('group1-shard1of1.bin', model_bin, s3_bucket)


if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=True)