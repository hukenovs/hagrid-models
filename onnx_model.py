import onnxruntime as ort

from hagrid_models.utils.onnx_utils import get_onnx_provider


class ONNXModel:
    def __init__(self, model_path, **kwargs):
        options, prov_opts, provider = get_onnx_provider()
        self.sess = ort.InferenceSession(
            model_path, sess_options=options, providers=provider, provider_options=prov_opts
        )
        self._get_input_output()

    def _get_input_output(self):
        inputs = self.sess.get_inputs()
        self.inputs = "".join(
            [
                f"\n {i}: {input.name}" f" Shape: ({','.join(map(str, input.shape))})" f" Dtype: {input.type}"
                for i, input in enumerate(inputs)
            ]
        )

        outputs = self.sess.get_outputs()
        self.outputs = "".join(
            [
                f"\n {i}: {output.name}" f" Shape: ({','.join(map(str, output.shape))})" f" Dtype: {output.type}"
                for i, output in enumerate(outputs)
            ]
        )

    def __repr__(self):
        return (
            f"Providers: {self.sess.get_providers()}\n"
            f"Model: {self.sess.get_modelmeta().description}\n"
            f"Version: {self.sess.get_modelmeta().version}\n"
            f"Inputs: {self.inputs}\n"
            f"Outputs: {self.outputs}"
        )

    def __call__(self, image, *args, **kwargs):
        outputs = self.sess.run(None, {"input": image})
        bboxes = outputs[0][0][..., :-1]
        scores = outputs[0][0][..., -1]
        labels = outputs[1][0]

        return {"boxes": bboxes, "scores": scores, "labels": labels}
