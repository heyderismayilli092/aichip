import os
import hashlib
import platform
import psutil
import pynvml
from pathlib import Path
import math
import gguf
import numpy as np
from gguf.constants import LlamaFileType


# displays information about the model in its operational state and the hardware on which it runs
class ModelInfo:
    def __init__(self, llm, model_path, n_ctx, n_gpu_layers):
        self.llm = llm
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers


    # enables NumPy data types to be returned in JSON format
    def json_safe(self, value):
        if isinstance(value, np.ndarray):
            return self.json_safe(value.tolist())

        if isinstance(value, np.generic):
            return self.json_safe(value.item())

        if isinstance(value, dict):
            return {str(k): self.json_safe(v) for k, v in value.items()}

        if isinstance(value, (list, tuple)):
            return [self.json_safe(v) for v in value]

        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")

        if isinstance(value, float) and not math.isfinite(value):
            return None

        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        return str(value)



    # retrieve model sha256sum
    def get_digest(self):
        sha256 = hashlib.sha256()
        with open(self.model_path, "rb") as file:
            for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


    # retrieve metadata
    def get_metadata(self):
        # load all metadata exposed by llama.cpp
        metadata = self.llm.metadata
        gguf_metadata = {value for key, value in metadata.items()}  # preserve all original metadata
        architecture = metadata.get("general.architecture")  # model architecture
        context_length = (metadata.get(f"{architecture}.context_length") if architecture else None)  # model context length

        # calculate parameter count from GGUF tensors
        reader = gguf.GGUFReader(str(self.model_path))
        parameter_count = sum(math.prod(tensor.shape) for tensor in reader.tensors)
        file_type = metadata.get("general.file_type")  # general file type

        # resolve quantization name
        try:
            quantization = (LlamaFileType(file_type).name if file_type is not None else None)
        except ValueError:
            quantization = f"UNKNOWN ({file_type})"

        # additional model information
        return {
            # derived information
            "format": self.model_path.suffix.lstrip("."),
            "family": architecture,
            "parameter_count": parameter_count,
            "parameter_size": f"{parameter_count / 1e9:.2f}B",
            "quantization_level": quantization,
            "model_context_length": context_length,
            "gguf_metadata": gguf_metadata  # original GGUF metadata
        }


    # retrieve GPU informations
    def get_gpu(self):
        try:
            pynvml.nvmlInit()
            devices = []

            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                devices.append({"name": pynvml.nvmlDeviceGetName(handle), "vram_total": memory.total,})

            pynvml.nvmlShutdown()
            return devices
        except Exception as error:
            return {"error": str(error)}


    # retrieve CPU informations
    def get_cpu(self):
        cpu_name = None
        try:
            with open("/proc/cpuinfo") as file:
                for line in file:
                    if line.startswith("model name"):
                        cpu_name = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass

        return {
            "name": cpu_name or platform.processor(),
            "architecture": platform.machine(),
            "logical_cores": psutil.cpu_count(),
            "physical_cores": psutil.cpu_count(logical=False)
        }


    # informations
    def snapshot(self):
        stat = self.model_path.stat()
        memory = psutil.virtual_memory()

        result = {
            "name": self.model_path.stem,
            "size": stat.st_size,
            "digest": self.get_digest(),
            "context_length": self.n_ctx,
            "gpu_layers": self.n_gpu_layers,
            "details": self.json_safe(self.get_metadata()),
            "hardware": {
                "cpu": self.get_cpu(),
                "ram": {
                    "total": memory.total,
                    "available": memory.available
                },
                "gpu": self.get_gpu()
            }
        }

        return result

