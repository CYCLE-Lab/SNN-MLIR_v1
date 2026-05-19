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
  // Reuse Conv shape helper logic.
  if (!hasShapeAndRank(getX()) || !hasShapeAndRank(getW()))
    return success();
  Type elementType = getElementType(getX().getType());
  ONNXConvOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
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
