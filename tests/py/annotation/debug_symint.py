"""Check that SymInt32 arithmetic produces valid grid expressions for KernelLaunchParams."""
import tensorrt as trt
import tensorrt.plugin as trtp

BLOCK = 256

@trtp.register("dbg::symtest")
def _desc(inp0: trtp.TensorDesc) -> trtp.TensorDesc:
    return inp0.like()

@trtp.aot_impl("dbg::symtest")
def _aot(inp0: trtp.TensorDesc, outputs: tuple[trtp.TensorDesc], tactic: int) -> tuple[str|bytes, str|bytes, trtp.KernelLaunchParams, trtp.SymIntExprs]:
    n = inp0.shape_expr[0]
    print(f"n type: {type(n)}")
    print(f"n: {n}")

    ceildiv = (n + (BLOCK - 1)) // BLOCK
    print(f"ceildiv type: {type(ceildiv)}")
    print(f"ceildiv: {ceildiv}")

    launch = trtp.KernelLaunchParams()
    launch.grid_x = ceildiv
    launch.grid_y = 1
    launch.grid_z = 1
    launch.block_x = 128
    launch.block_y = 1
    launch.block_z = 1
    launch.shared_mem = 0

    extra = trtp.SymIntExprs(1)
    extra[0] = n
    print(f"extra[0] type: {type(extra[0])}")

    return b"dummy_kernel", b"dummy_ptx", launch, extra


logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network()
inp = network.add_input("x", trt.float32, (1024,))

inputs, shape_inputs, plugin = trtp.op.dbg.symtest(inp)(trt.QuickPluginCreationRequest.STRICT_AOT)
layer = network.add_plugin_v3(inputs, shape_inputs, plugin)
if layer is not None:
    print("Layer added successfully, output:", layer.get_output(0).name)
else:
    print("Layer is None")
