from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
from torch.fx.node import Node
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._ConverterRegistry import ConverterPriority
from torch_tensorrt.dynamo.conversion.plugins._generate_plugin import generate_plugin
from torch_tensorrt.dynamo.conversion.plugins._generate_plugin_converter import (
    generate_plugin_converter,
)

if TYPE_CHECKING:
    from torch_tensorrt.annotation._custom_plugin._descriptor import CustomPluginSpec


def custom_op(
    op_name: str,
    impl: Optional["CustomPluginSpec"] = None,
    capability_validator: Optional[Callable[[Node, CompilationSettings], bool]] = None,
    priority: ConverterPriority = ConverterPriority.STANDARD,
    supports_dynamic_shapes: bool = False,
    requires_output_allocator: bool = False,
) -> None:
    """
    Generate the Plugin and corresponding Plugin Converter using external kernels
    and TensorRT Quick Deployable Plugin APIs.

    Args:
        op_name: the plugin name in ``"namespace::name"`` form.  A matching
            ``torch.library.custom_op`` must already exist (or be auto-created
            via ``impl``).
        impl: optional ``tta.CustomPluginSpec`` from ``tta.custom_plugin(...)``.
            When provided:

            1. The torch op is auto-registered (``impl.auto_register_torch_op``).
            2. TRT QDP descriptors are registered via ``register_custom_plugin``
               which handles both activation and weight tensor inputs, then adds
               ``@trtp.autotune`` and ``@trtp.aot_impl`` for AOT kernel dispatch.
            3. A Dynamo converter is registered that calls ``impl.lower_to_trt``
               so that weight constants are injected as ``trt.add_constant`` layers
               before the plugin sees them, mirroring the native TTA lowering path.

            When ``None`` (default) the existing JIT path (``generate_plugin`` +
            ``generate_plugin_converter``) is used unchanged.
        capability_validator: optional node capability predicate.
        priority: converter registry priority.
        supports_dynamic_shapes: whether the converter supports dynamic shapes.
        requires_output_allocator: whether the converter requires an output
            allocator (e.g. data-dependent operators).
    """
    if impl is not None:
        from torch_tensorrt.annotation._custom_plugin._descriptor import (
            register_custom_plugin,
        )
        from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
            dynamo_tensorrt_converter,
        )
        from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor

        impl.auto_register_torch_op(op_name)

        namespace, op_local_name = op_name.split("::")
        torch_op = getattr(getattr(torch.ops, namespace), op_local_name)
        schema = torch_op._schemas[""]
        n_tensor_inputs = sum(
            1 for a in schema.arguments
            if a.type.isSubtypeOf(torch._C.TensorType.get())
        )

        # QDP registration: includes weight tensors in the input count so TRT
        # can receive them as trt.add_constant outputs.
        register_custom_plugin(
            impl,
            num_inputs=n_tensor_inputs + len(impl.weights),
            num_outputs=impl.num_outputs,
            qdp_name=op_name,
        )

        torch_overload = getattr(torch_op, "default")
        _impl = impl
        _n_act = n_tensor_inputs
        _qdp_name = op_name

        def _impl_converter(
            ctx: Any,
            target: Any,
            args: Any,
            kwargs: Any,
            name: str,
        ) -> Any:
            unique_id = uuid.uuid4()
            itensor_args = [
                get_trt_tensor(ctx, t, f"inp{i}_{unique_id}")
                for i, t in enumerate(args[:_n_act])
            ]
            return _impl.lower_to_trt(ctx, itensor_args, name, qdp_name=_qdp_name)

        dynamo_tensorrt_converter(
            torch_overload,
            capability_validator=capability_validator,
            priority=priority,
            supports_dynamic_shapes=supports_dynamic_shapes,
            requires_output_allocator=requires_output_allocator,
        )(_impl_converter)
    else:
        generate_plugin(op_name)
        generate_plugin_converter(
            op_name,
            capability_validator,
            priority,
            supports_dynamic_shapes,
            requires_output_allocator,
        )
