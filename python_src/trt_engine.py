"""Инференс self-play через TensorRT.

Зачем: замер 14.09 на этой сети (47M, 384 канала) дал 1.40x на батче 96 и 1.50x
на 256 против нынешнего пути (копия весов в BF16 + torch.compile), при полном
совпадении лучшего хода со 100% позиций и расхождении оценки 0.0008. TensorRT
собирает граф заранее, перебирая реализации каждого слоя на этой конкретной
карте, и сращивает мелкие операции — а мы упираемся именно в их количество.

Движок пересобирается каждую итерацию: веса меняются после обучения, а
UltraFastMCTS создаётся заново в generate_games, так что отдельного управления
временем жизни не нужно.

Точность: граф идёт в FP16. FP8 и FP4 проверены и отвергнуты — 1.19x при ошибке
оценки 0.035 (в сорок раз хуже FP16), см. project_vram_384ch и разбор в HANDOVER.
"""
import os
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn


class _RawOutputs(nn.Module):
    """Только то, что потребляет поиск: логиты политики, WDL и moves-left.

    Голова future — обучающая, в поиске не нужна; лишние выходы мешают TensorRT
    сращивать граф."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        if isinstance(out, tuple) and len(out) >= 3:
            return out[0], out[1], out[2]
        if isinstance(out, tuple) and len(out) == 2:
            return out[0], out[1]
        return out


class TRTNet:
    """Обёртка вокруг движка TensorRT, неотличимая для поиска от обычной сети:
    принимает (B, C, H, W) и возвращает кортеж логитов."""

    def __init__(self, net: nn.Module, device, max_batch: int,
                 workspace_gb: float = 2.0, log=print):
        import tensorrt as trt                      # импорт здесь: не нужен, если TRT не включён
        self._trt = trt
        self._logger = trt.Logger(trt.Logger.WARNING)
        self.device = device
        self.max_batch = int(max_batch)

        src = net._orig_mod if hasattr(net, "_orig_mod") else net
        # Экспорт идёт с ПРОЦЕССОРА: на видеокарте RMSNorm исполняется слитым
        # ядром aten._fused_rms_norm, для которого в ONNX нет представления и
        # экспорт падает. На процессоре оно раскладывается на примитивы.
        import copy as _copy
        wrapper = _RawOutputs(_copy.deepcopy(src).float().cpu()).eval()

        t0 = time.time()
        onnx_path = self._export(wrapper)
        t_export = time.time() - t0

        t0 = time.time()
        self._engine = self._build(onnx_path, workspace_gb)
        t_build = time.time() - t0
        os.unlink(onnx_path)

        self._ctx = self._engine.create_execution_context()
        self._in_name = self._engine.get_tensor_name(0)
        self._out_names = [self._engine.get_tensor_name(i)
                           for i in range(self._engine.num_io_tensors)
                           if self._engine.get_tensor_mode(
                               self._engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
        self._dtypes = {trt.float32: torch.float32, trt.float16: torch.float16}
        self.in_dtype = self._dtypes[self._engine.get_tensor_dtype(self._in_name)]
        self._stream = torch.cuda.Stream()
        self._bufs = {}
        log(f"   TensorRT: движок собран (экспорт {t_export:.0f} с, сборка {t_build:.0f} с, "
            f"батч до {self.max_batch})")

    def _export(self, wrapper):
        """FP32-граф средствами torch, затем перевод в FP16.

        Прямой экспорт половинной точности из torch на этой модели падает
        («Translate the graph into ONNX ❌»), а перевод готового графа проходит."""
        import onnx
        from onnxconverter_common import float16
        from model import CapablancaNet

        fd, path = tempfile.mkstemp(suffix=".onnx")
        os.close(fd)
        dummy = torch.zeros(2, CapablancaNet.INPUT_PLANES, CapablancaNet.BOARD_H,
                            CapablancaNet.BOARD_W)
        with torch.no_grad():
            torch.onnx.export(wrapper, (dummy,), path, input_names=["board"],
                              output_names=["policy", "wdl", "mlh"],
                              opset_version=18, dynamic_axes={"board": {0: "batch"}},
                              dynamo=True)
        m16 = float16.convert_float_to_float16(onnx.load(path), keep_io_types=False,
                                               disable_shape_infer=True)
        onnx.save(m16, path)
        return path

    def _build(self, onnx_path, workspace_gb):
        trt = self._trt
        b = trt.Builder(self._logger)
        net = b.create_network(0)
        parser = trt.OnnxParser(net, self._logger)
        if not parser.parse(open(onnx_path, "rb").read()):
            errs = "; ".join(str(parser.get_error(i)) for i in range(min(parser.num_errors, 3)))
            raise RuntimeError(f"TensorRT не разобрал граф: {errs}")
        cfg = b.create_builder_config()
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
        inp = net.get_input(0)
        c, h, w = inp.shape[1], inp.shape[2], inp.shape[3]
        prof = b.create_optimization_profile()
        # Поиск подаёт батчи разной длины (адаптивные бакеты плюс добивка
        # последней пачки), поэтому профиль от единицы до потолка.
        prof.set_shape(inp.name, (1, c, h, w), (self.max_batch, c, h, w),
                       (self.max_batch, c, h, w))
        cfg.add_optimization_profile(prof)
        plan = b.build_serialized_network(net, cfg)
        if plan is None:
            raise RuntimeError("TensorRT не собрал движок")
        return trt.Runtime(self._logger).deserialize_cuda_engine(bytes(plan))

    def _out_buf(self, name, shape):
        key = (name, shape)
        buf = self._bufs.get(key)
        if buf is None:
            buf = torch.empty(shape, dtype=self._dtypes[self._engine.get_tensor_dtype(name)],
                              device=self.device)
            self._bufs[key] = buf
        return buf

    def __call__(self, x: torch.Tensor):
        if x.shape[0] > self.max_batch:
            raise RuntimeError(f"батч {x.shape[0]} больше потолка движка {self.max_batch}")
        x = x.to(self.in_dtype).contiguous()
        self._ctx.set_input_shape(self._in_name, tuple(x.shape))
        self._ctx.set_tensor_address(self._in_name, x.data_ptr())
        outs = []
        for name in self._out_names:
            buf = self._out_buf(name, tuple(self._ctx.get_tensor_shape(name)))
            self._ctx.set_tensor_address(name, buf.data_ptr())
            outs.append(buf)
        self._ctx.execute_async_v3(self._stream.cuda_stream)
        torch.cuda.current_stream().wait_stream(self._stream)
        return tuple(outs)

    def eval(self):
        return self

    def train(self, mode=True):
        return self
