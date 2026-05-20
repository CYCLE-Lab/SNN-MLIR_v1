#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Operation.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include <functional>

using namespace mlir;

#define DEBUG_TYPE "lower-memref-reinterpret-cast"

namespace {

class LowerMemRefReinterpretCastPass
    : public PassWrapper<LowerMemRefReinterpretCastPass,
                         OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const final {
    return "lower-memref-reinterpret-cast";
  }

  StringRef getDescription() const final {
    return "Materialize memref.reinterpret_cast into explicit affine.for loops "
           "and memref.alloca buffers.";
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();
    OpBuilder builder(context);

    SmallVector<memref::ReinterpretCastOp> reinterpretOps;
    moduleOp.walk([&](memref::ReinterpretCastOp op) {
      reinterpretOps.push_back(op);
    });

    for (memref::ReinterpretCastOp reinterpretOp : reinterpretOps) {
      if (failed(lowerOneReinterpretCast(builder, reinterpretOp))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  static bool hasStaticShape(MemRefType type) {
    if (!type.hasStaticShape())
      return false;
    return true;
  }

  static SmallVector<int64_t> computeIdentityStrides(MemRefType type) {
    SmallVector<int64_t> strides(type.getRank(), 1);

    int64_t running = 1;
    for (int64_t i = type.getRank() - 1; i >= 0; --i) {
      strides[i] = running;
      running *= type.getDimSize(i);
    }

    return strides;
  }

  LogicalResult lowerOneReinterpretCast(
      OpBuilder &builder, memref::ReinterpretCastOp reinterpretOp) {
    Location loc = reinterpretOp.getLoc();

    Value source = reinterpretOp.getSource();
    auto sourceType = source.getType().dyn_cast<MemRefType>();
    auto resultType = reinterpretOp.getResult().getType().dyn_cast<MemRefType>();

    if (!sourceType || !resultType) {
      reinterpretOp.emitError("expects memref source and memref result");
      return failure();
    }

    if (!hasStaticShape(sourceType) || !hasStaticShape(resultType)) {
      reinterpretOp.emitError(
          "only static memref shapes are supported by this pass");
      return failure();
    }

    ArrayRef<int64_t> targetSizes = reinterpretOp.getStaticSizes();
    ArrayRef<int64_t> targetStrides = reinterpretOp.getStaticStrides();
    ArrayRef<int64_t> targetOffsets = reinterpretOp.getStaticOffsets();

    if (targetSizes.size() != static_cast<size_t>(resultType.getRank()) ||
        targetStrides.size() != static_cast<size_t>(resultType.getRank()) ||
        targetOffsets.size() != 1) {
      reinterpretOp.emitError("unexpected reinterpret_cast metadata");
      return failure();
    }

    int64_t targetOffset = targetOffsets[0];

    if (ShapedType::isDynamic(targetOffset)) {
      reinterpretOp.emitError("dynamic offset is not supported");
      return failure();
    }

    for (int64_t size : targetSizes) {
      if (ShapedType::isDynamic(size)) {
        reinterpretOp.emitError("dynamic size is not supported");
        return failure();
      }
    }

    for (int64_t stride : targetStrides) {
      if (ShapedType::isDynamic(stride)) {
        reinterpretOp.emitError("dynamic stride is not supported");
        return failure();
      }
    }

    for (int64_t i = 0; i < resultType.getRank(); ++i) {
      if (resultType.getDimSize(i) != targetSizes[i]) {
        reinterpretOp.emitError(
            "result memref shape does not match reinterpret_cast sizes");
        return failure();
      }
    }

    SmallVector<int64_t> sourceStrides = computeIdentityStrides(sourceType);

    // 1. 在 reinterpret_cast 前创建物化 buffer。
    builder.setInsertionPoint(reinterpretOp);
    auto materialized = builder.create<memref::AllocaOp>(
        loc, resultType);

    // 2. 在 reinterpret_cast 后插入 affine loop，填充 materialized。
    builder.setInsertionPointAfter(reinterpretOp);

    SmallVector<Value> targetIvs;

    std::function<void(unsigned)> buildLoops = [&](unsigned depth) {
      if (depth == static_cast<unsigned>(resultType.getRank())) {
        emitCopyElement(builder, loc, source, sourceType,
                        materialized.getResult(), resultType,
                        targetIvs, targetOffset, targetStrides,
                        sourceStrides);
        return;
      }

      int64_t ub = targetSizes[depth];

      auto forOp = builder.create<affine::AffineForOp>(
          loc,
          /*lowerBound=*/0,
          /*upperBound=*/ub,
          /*step=*/1);

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(forOp.getBody());

      targetIvs.push_back(forOp.getInductionVar());
      buildLoops(depth + 1);
      targetIvs.pop_back();
    };

    buildLoops(/*depth=*/0);

    // 3. 替换所有使用。
    reinterpretOp.getResult().replaceAllUsesWith(materialized.getResult());

    // 4. 删除原始 reinterpret_cast。
    reinterpretOp.erase();

    return success();
  }

  void emitCopyElement(OpBuilder &builder,
                       Location loc,
                       Value source,
                       MemRefType sourceType,
                       Value target,
                       MemRefType targetType,
                       ValueRange targetIvs,
                       int64_t targetOffset,
                       ArrayRef<int64_t> targetStrides,
                       ArrayRef<int64_t> sourceStrides) {
    MLIRContext *context = builder.getContext();

    unsigned targetRank = targetType.getRank();
    unsigned sourceRank = sourceType.getRank();

    SmallVector<AffineExpr> dims;
    dims.reserve(targetRank);
    for (unsigned i = 0; i < targetRank; ++i)
      dims.push_back(builder.getAffineDimExpr(i));

    AffineExpr linearExpr = builder.getAffineConstantExpr(targetOffset);
    for (unsigned i = 0; i < targetRank; ++i) {
      linearExpr = linearExpr + dims[i] * targetStrides[i];
    }

    SmallVector<Value> sourceIndices;
    sourceIndices.reserve(sourceRank);

    for (unsigned i = 0; i < sourceRank; ++i) {
      AffineExpr idxExpr = linearExpr.floorDiv(sourceStrides[i]);

      // 对非最高维做取模，得到该维度的真实 index。
      //
      // 例如 source memref<4x40x98xf32>：
      //   source strides = [3920, 98, 1]
      //
      // linear = i0 * 3920 + i1 * 3920 + i2 * 98 + i3
      //
      // source index:
      //   d0 = linear floordiv 3920
      //   d1 = (linear floordiv 98) mod 40
      //   d2 = linear mod 98
      if (i + 1 < sourceRank) {
        int64_t dimSize = sourceType.getDimSize(i);
        idxExpr = idxExpr % dimSize;
      }

      auto map = AffineMap::get(
          /*dimCount=*/targetRank,
          /*symbolCount=*/0,
          idxExpr,
          context);

      Value idx = builder.create<affine::AffineApplyOp>(
          loc, map, targetIvs);

      sourceIndices.push_back(idx);
    }

    Value loaded = builder.create<memref::LoadOp>(
        loc, source, sourceIndices);

    builder.create<memref::StoreOp>(
        loc, loaded, target, targetIvs);
  }
};

} // namespace

namespace onnx_mlir {
std::unique_ptr<Pass> createLowerMemRefReinterpretCastPass() {
  return std::make_unique<LowerMemRefReinterpretCastPass>();
}
} // namespace onnx_mlir

static PassRegistration<LowerMemRefReinterpretCastPass> pass;