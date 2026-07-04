#if canImport(Accelerate)
@_exported import Accelerate
#else
import CBLAS
import Glibc

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

// MARK: - Single-precision Level-1 CBLAS shims
//
// Pure-Swift Level-1 routines (Int32 signatures matching the consumers). Scalar
// loops rather than @_extern OpenBLAS bindings to sidestep the CBLAS_INDEX/size_t
// return-ABI of `cblas_isamax`; correctness over peak throughput.

/// Single-precision dot product.
@inlinable
public func cblas_sdot(_ n: Int32, _ x: UnsafePointer<Float>, _ incX: Int32, _ y: UnsafePointer<Float>, _ incY: Int32) -> Float {
    var sum: Float = 0
    var i = 0
    while i < Int(n) { sum += x[i * Int(incX)] * y[i * Int(incY)]; i += 1 }
    return sum
}

/// Single-precision scale in place: x = alpha * x.
@inlinable
public func cblas_sscal(_ n: Int32, _ alpha: Float, _ x: UnsafeMutablePointer<Float>, _ incX: Int32) {
    var i = 0
    while i < Int(n) { x[i * Int(incX)] *= alpha; i += 1 }
}

/// Single-precision AXPY: y = alpha * x + y.
@inlinable
public func cblas_saxpy(_ n: Int32, _ alpha: Float, _ x: UnsafePointer<Float>, _ incX: Int32, _ y: UnsafeMutablePointer<Float>, _ incY: Int32) {
    var i = 0
    while i < Int(n) { y[i * Int(incY)] += alpha * x[i * Int(incX)]; i += 1 }
}

/// Single-precision Euclidean norm: sqrt(sum(x^2)).
@inlinable
public func cblas_snrm2(_ n: Int32, _ x: UnsafePointer<Float>, _ incX: Int32) -> Float {
    var sum: Float = 0
    var i = 0
    while i < Int(n) { let v = x[i * Int(incX)]; sum += v * v; i += 1 }
    return sqrtf(sum)
}

/// Single-precision sum of absolute values.
@inlinable
public func cblas_sasum(_ n: Int32, _ x: UnsafePointer<Float>, _ incX: Int32) -> Float {
    var sum: Float = 0
    var i = 0
    while i < Int(n) { sum += abs(x[i * Int(incX)]); i += 1 }
    return sum
}

/// Single-precision index of the element with the largest absolute value.
@inlinable
public func cblas_isamax(_ n: Int32, _ x: UnsafePointer<Float>, _ incX: Int32) -> Int32 {
    guard n > 0 else { return 0 }
    var best = 0
    var m = abs(x[0])
    var i = 1
    while i < Int(n) {
        let v = abs(x[i * Int(incX)])
        if v > m { m = v; best = i }
        i += 1
    }
    return Int32(best)
}
#endif  // canImport(Accelerate)
