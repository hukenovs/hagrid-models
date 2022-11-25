import onnxruntime as ort


def get_onnx_provider():
    provider = ["CPUExecutionProvider"]
    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    prov_opts = []
    print("Using ONNX Runtime", ort.get_device())

    if "DML" in ort.get_device():
        prov_opts = [{"device_id": 0}]
        provider.append("DmlExecutionProvider")

    elif "GPU" in ort.get_device():
        prov_opts = [
            {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            }
        ]
        provider.append("CUDAExecutionProvider")

    return options, prov_opts, provider
