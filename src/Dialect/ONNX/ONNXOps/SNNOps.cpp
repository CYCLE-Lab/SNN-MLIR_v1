/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ SNN Ops - shape inference ------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// LIF
//===----------------------------------------------------------------------===//

LogicalResult ONNXLIFOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Need both x and v_combined shapes to infer outputs.
  if (!hasShapeAndRank(getX()) || !hasShapeAndRank(getVCombined()))
    return success();

  if (auto xTy = dyn_cast<ShapedType>(getX().getType())) {
    updateType(getOperation(), getSpike(), xTy.getShape(),
        xTy.getElementType());
  }

  if (auto vTy = dyn_cast<ShapedType>(getVCombined().getType())) {
    updateType(getOperation(), getVCombinedNext(), vTy.getShape(),
        vTy.getElementType());
  }
  return success();
}

//===----------------------------------------------------------------------===//
// MultiStepLIF
//===----------------------------------------------------------------------===//

LogicalResult ONNXMultiStepLIFOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getXSeq()) || !hasShapeAndRank(getVCombinedInit()))
    return success();

  if (auto xTy = dyn_cast<ShapedType>(getXSeq().getType())) {
    updateType(getOperation(), getSpikeSeq(), xTy.getShape(),
        xTy.getElementType());
  }

  if (auto vTy = dyn_cast<ShapedType>(getVCombinedInit().getType())) {
    updateType(getOperation(), getVCombinedFinal(), vTy.getShape(),
        vTy.getElementType());
  }
  return success();
}

//===----------------------------------------------------------------------===//
// DataToVector
//===----------------------------------------------------------------------===//

LogicalResult ONNXDataToVectorOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getInput()))
    return success();

  auto inTy = dyn_cast<ShapedType>(getInput().getType());
  if (!inTy)
    return success();

  SmallVector<int64_t> outShape;
  auto inShape = inTy.getShape();
  if (inShape.size() == 0) // unexpected
    return success();

  for (size_t i = 0; i + 1 < inShape.size(); ++i)
    outShape.push_back(inShape[i]);

  int64_t last = inShape.back();
  int64_t packed = ShapedType::kDynamic;
  if (last != ShapedType::kDynamic) {
    int64_t bitWidth = 32;
    if (auto bwAttr = getBitWidthAttr())
      bitWidth = bwAttr.getInt();
    if (bitWidth > 0)
      packed = (last + bitWidth - 1) / bitWidth;
  }
  outShape.push_back(packed);

  // Use 32-bit integer as packed representation element type.
  Type elt = IntegerType::get(getContext(), 32);
  updateType(getOperation(), getOutputVector(), outShape, elt);
  return success();
}

//===----------------------------------------------------------------------===//
// SNNFC
//===----------------------------------------------------------------------===//

LogicalResult ONNXSNNFCOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // The op carries a shape_anchor operand whose type is used as the output
  // shape/type. If not present or not ranked, nothing to do.
  if (!hasShapeAndRank(getShapeAnchor()))
    return success();

  if (auto anchorTy = dyn_cast<ShapedType>(getShapeAnchor().getType())) {
    updateType(getOperation(), getX(), anchorTy.getShape(),
        anchorTy.getElementType());
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SConv
//===----------------------------------------------------------------------===//

LogicalResult ONNXSConvOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getX()) || !hasShapeAndRank(getW()))
    return success();

  // ONNXConvOpShapeHelper cannot be reused here: its computeShape()
  // specialization unconditionally casts the operation to ONNXConvOp.  SConv
  // has the same spatial formula but is a distinct operation class.
  auto xTy = dyn_cast<RankedTensorType>(getX().getType());
  auto wTy = dyn_cast<RankedTensorType>(getW().getType());
  if (!xTy || !wTy)
    return success();
  if (xTy.getRank() != wTy.getRank() || xTy.getRank() < 3)
    return emitError("SConv input and weight must have equal rank >= 3");

  const int64_t spatialRank = xTy.getRank() - 2;
  auto kernelShape = getKernelShape();
  auto pads = getPads();
  auto strides = getStrides();
  auto dilations = getDilations();
  if (kernelShape && static_cast<int64_t>(kernelShape->size()) != spatialRank)
    return emitError("SConv kernel_shape rank mismatch");
  if (pads && static_cast<int64_t>(pads->size()) != 2 * spatialRank)
    return emitError("SConv pads rank mismatch");
  if (strides && static_cast<int64_t>(strides->size()) != spatialRank)
    return emitError("SConv strides rank mismatch");
  if (dilations && static_cast<int64_t>(dilations->size()) != spatialRank)
    return emitError("SConv dilations rank mismatch");
  if (getAutoPad() != "NOTSET" && getAutoPad() != "VALID")
    return emitError("SConv shape inference supports NOTSET or VALID auto_pad");

  auto attrValue = [](std::optional<ArrayAttr> attr, int64_t index,
                       int64_t defaultValue) {
    if (!attr)
      return defaultValue;
    return cast<IntegerAttr>((*attr)[index]).getInt();
  };

  SmallVector<int64_t, 4> outputShape;
  outputShape.emplace_back(xTy.getDimSize(0));
  outputShape.emplace_back(wTy.getDimSize(0));
  for (int64_t i = 0; i < spatialRank; ++i) {
    const int64_t input = xTy.getDimSize(i + 2);
    const int64_t kernel = kernelShape
                               ? attrValue(kernelShape, i, 0)
                               : wTy.getDimSize(i + 2);
    const int64_t stride = attrValue(strides, i, 1);
    const int64_t dilation = attrValue(dilations, i, 1);
    const int64_t padBegin = getAutoPad() == "VALID"
                                 ? 0
                                 : attrValue(pads, i, 0);
    const int64_t padEnd = getAutoPad() == "VALID"
                               ? 0
                               : attrValue(pads, i + spatialRank, 0);
    if (stride <= 0 || dilation <= 0 || kernel <= 0)
      return emitError("SConv kernel, stride, and dilation must be positive");
    if (ShapedType::isDynamic(input) || ShapedType::isDynamic(kernel)) {
      outputShape.emplace_back(ShapedType::kDynamic);
      continue;
    }
    const int64_t effectiveKernel = dilation * (kernel - 1) + 1;
    outputShape.emplace_back(
        (input + padBegin + padEnd - effectiveKernel) / stride + 1);
  }

  updateType(getOperation(), getY(), outputShape, xTy.getElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// IF
//===----------------------------------------------------------------------===//

LogicalResult ONNXIFOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getX()))
    return success();

  // spike has same shape as x
  if (auto xTy = dyn_cast<ShapedType>(getX().getType())) {
    updateType(getOperation(), getSpike(), xTy.getShape(),
        xTy.getElementType());
    // v_next: same shape as x (or v if provided and ranked)
    if (hasShapeAndRank(getV())) {
      if (auto vTy = dyn_cast<ShapedType>(getV().getType()))
        updateType(getOperation(), getVNext(), vTy.getShape(),
            vTy.getElementType());
      else
        updateType(getOperation(), getVNext(), xTy.getShape(),
            xTy.getElementType());
    } else {
      updateType(getOperation(), getVNext(), xTy.getShape(),
          xTy.getElementType());
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// MultiStepIF
//===----------------------------------------------------------------------===//

LogicalResult ONNXMultiStepIFOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getXSeq()))
    return success();

  if (auto xTy = dyn_cast<ShapedType>(getXSeq().getType())) {
    updateType(getOperation(), getSpikeSeq(), xTy.getShape(),
        xTy.getElementType());
  }

  if (hasShapeAndRank(getVInit())) {
    if (auto vTy = dyn_cast<ShapedType>(getVInit().getType()))
      updateType(getOperation(), getVFinal(), vTy.getShape(),
          vTy.getElementType());
  }
  return success();
}
