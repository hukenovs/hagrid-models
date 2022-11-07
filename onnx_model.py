from utils.onnx_utils import get_onnx_provider
import numpy as np
import onnxruntime as ort


class ONNXModel:
    def __init__(self, model_path, **kwargs):
        options, prov_opts, provider = get_onnx_provider()
        self.sess = ort.InferenceSession(model_path, sess_options=options,
                                         providers=provider, provider_options=prov_opts)
        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])
        self._get_input_output()

    def _get_input_output(self):
        inputs = self.sess.get_inputs()
        self.inputs = "".join([f"\n {i}: {input.name}"
                               f" Shape: ({','.join(map(str, input.shape))})"
                               f" Dtype: {input.type}" for i, input in enumerate(inputs)])

        outputs = self.sess.get_outputs()
        self.outputs = "".join([f"\n {i}: {output.name}"
                                f" Shape: ({','.join(map(str, output.shape))})"
                                f" Dtype: {output.type}" for i, output in enumerate(outputs)])

    def __repr__(self):
        return f"Providers: {self.sess.get_providers()}\n" \
               f"Model: {self.sess.get_modelmeta().description}\n" \
               f"Version: {self.sess.get_modelmeta().version}\n" \
               f"Inputs: {self.inputs}\n" \
               f"Outputs: {self.outputs}"

    def _transform(self, image):
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        return image

    def __call__(self, image, *args, **kwargs):
        input_data = self._transform(image)
        outputs = self.sess.run(None, {"input": input_data})
        bboxes = outputs[0][..., :-1]
        scores = outputs[0][..., -1]
        labels = outputs[1]

        return bboxes, scores, labels


