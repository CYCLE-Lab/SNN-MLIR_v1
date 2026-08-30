// RUN: onnx-mlir-opt --shape-inference %s | FileCheck %s

func.func @sconv_shape(%x: tensor<1x3x64x64xf32>,
                       %w: tensor<16x3x3x3xf32>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.SConv"(%x, %w, %none) {
    auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
    kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]
  } : (tensor<1x3x64x64xf32>, tensor<16x3x3x3xf32>, none) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// CHECK-LABEL: func.func @sconv_shape
// CHECK: "onnx.SConv"{{.*}} -> tensor<1x16x64x64xf32>
// CHECK: onnx.Return {{.*}} : tensor<1x16x64x64xf32>
