#if canImport(Accelerate)
@_exported import Accelerate
#else
import CBLAS

// MARK: - Single-precision (Float) CBLAS shims
//
// Mirror the double-precision `cblas_dgemm` `@_extern` declaration in `BLAS.swift`
// for the single-precision routines used by single-precision consumers. These
// bind directly to the OpenBLAS C symbols at link time (the package links
// "openblas"); no Swift implementation is provided.

/// Single-precision general matrix-matrix multiply: C = alpha*op(A)*op(B) + beta*C.
@inlinable
@_extern(c, "cblas_sgemm")
public func cblas_sgemm(
    _ __Order: CBLAS_ORDER,
    _ __TransA: CBLAS_TRANSPOSE,
    _ __TransB: CBLAS_TRANSPOSE,
    _ __M: blasint,
    _ __N: blasint,
    _ __K: blasint,
    _ __alpha: Float,
    _ __A: UnsafePointer<Float>!,
    _ __lda: blasint,
    _ __B: UnsafePointer<Float>!,
    _ __ldb: blasint,
    _ __beta: Float,
    _ __C: UnsafeMutablePointer<Float>!,
    _ __ldc: blasint
)

/// Single-precision general matrix-vector multiply: Y = alpha*op(A)*X + beta*Y.
@inlinable
@_extern(c, "cblas_sgemv")
public func cblas_sgemv(
    _ __Order: CBLAS_ORDER,
    _ __TransA: CBLAS_TRANSPOSE,
    _ __M: blasint,
    _ __N: blasint,
    _ __alpha: Float,
    _ __A: UnsafePointer<Float>!,
    _ __lda: blasint,
    _ __X: UnsafePointer<Float>!,
    _ __incX: blasint,
    _ __beta: Float,
    _ __Y: UnsafeMutablePointer<Float>!,
    _ __incY: blasint
)

/// Single-precision rank-1 update: A = alpha*X*Y^T + A.
@inlinable
@_extern(c, "cblas_sger")
public func cblas_sger(
    _ __Order: CBLAS_ORDER,
    _ __M: blasint,
    _ __N: blasint,
    _ __alpha: Float,
    _ __X: UnsafePointer<Float>!,
    _ __incX: blasint,
    _ __Y: UnsafePointer<Float>!,
    _ __incY: blasint,
    _ __A: UnsafeMutablePointer<Float>!,
    _ __lda: blasint
)
#endif  // canImport(Accelerate)
