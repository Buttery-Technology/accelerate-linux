#if canImport(Accelerate)
@_exported import Accelerate
#else
import Glibc

// MARK: - Single-precision (Float) vDSP shims
//
// Linux does not ship Apple's Accelerate framework. This package provides
// pure-Swift fallbacks; the upstream package supplied only the double-precision
// (`*D`) surface. These single-precision variants mirror the corresponding
// double-precision implementations in `VectorBasicOps.swift` so that
// single-precision consumers (e.g. ValueStore ML, KnowledgeStoreKit PCA,
// QueryDecomposer) compile and run on Linux.
//
// Argument order for `vDSP_vsub` / `vDSP_vdiv` intentionally matches Accelerate
// (subtrahend/divisor first). Behavior mirrors the documented Accelerate
// semantics; these scalar loops are not SIMD-optimized.

/// Single-precision element-wise sum: C = A + B.
@inlinable
public func vDSP_vadd(
    _ __A: UnsafePointer<Float>,
    _ __IA: vDSP_Stride,
    _ __B: UnsafePointer<Float>,
    _ __IB: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Float>,
    _ __IC: vDSP_Stride,
    _ __N: vDSP_Length
) {
    var i = 0
    while i < __N {
        __C[i * __IC] = __A[i * __IA] + __B[i * __IB]
        i += 1
    }
}

/// Single-precision element-wise difference: C = A - B (B is the first argument).
@inlinable
public func vDSP_vsub(
    _ __B: UnsafePointer<Float>,
    _ __IB: vDSP_Stride,
    _ __A: UnsafePointer<Float>,
    _ __IA: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Float>,
    _ __IC: vDSP_Stride,
    _ __N: vDSP_Length
) {
    var i = 0
    while i < __N {
        __C[i * __IC] = __A[i * __IA] - __B[i * __IB]
        i += 1
    }
}

/// Single-precision element-wise product: C = A * B.
@inlinable
public func vDSP_vmul(
    _ __A: UnsafePointer<Float>,
    _ __IA: vDSP_Stride,
    _ __B: UnsafePointer<Float>,
    _ __IB: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Float>,
    _ __IC: vDSP_Stride,
    _ __N: vDSP_Length
) {
    var i = 0
    while i < __N {
        __C[i * __IC] = __A[i * __IA] * __B[i * __IB]
        i += 1
    }
}

/// Single-precision element-wise division: C = A / B (B is the first argument).
@inlinable
public func vDSP_vdiv(
    _ __B: UnsafePointer<Float>,
    _ __IB: vDSP_Stride,
    _ __A: UnsafePointer<Float>,
    _ __IA: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Float>,
    _ __IC: vDSP_Stride,
    _ __N: vDSP_Length
) {
    var i = 0
    while i < __N {
        __C[i * __IC] = __A[i * __IA] / __B[i * __IB]
        i += 1
    }
}

/// Single-precision division by a scalar: C = A / (*B).
@inlinable
public func vDSP_vsdiv(
    _ __A: UnsafePointer<Float>,
    _ __IA: vDSP_Stride,
    _ __B: UnsafePointer<Float>,
    _ __C: UnsafeMutablePointer<Float>,
    _ __IC: vDSP_Stride,
    _ __N: vDSP_Length
) {
    var i = 0
    while i < __N {
        __C[i * __IC] = __A[i * __IA] / __B.pointee
        i += 1
    }
}

/// Single-precision absolute value: C = |A|.
@inlinable
public func vDSP_vabs(
    _ __A: UnsafePointer<Float>,
    _ __IA: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Float>,
    _ __IC: vDSP_Stride,
    _ __N: vDSP_Length
) {
    var i = 0
    while i < __N {
        __C[i * __IC] = abs(__A[i * __IA])
        i += 1
    }
}

/// Single-precision vector limit: D = (*C) if A >= (*B), else -(*C).
@inlinable
public func vDSP_vlim(
    _ __A: UnsafePointer<Float>,
    _ __IA: vDSP_Stride,
    _ __B: UnsafePointer<Float>,
    _ __C: UnsafePointer<Float>,
    _ __D: UnsafeMutablePointer<Float>,
    _ __ID: vDSP_Stride,
    _ __N: vDSP_Length
) {
    var i = 0
    while i < __N {
        __D[i * __ID] = __C.pointee * ((__B.pointee <= __A[i * __IA]) ? 1 : -1)
        i += 1
    }
}

/// Single-precision multiply by scalar: C = A * (*B).
@inlinable
public func vDSP_vsmul(
    _ __A: UnsafePointer<Float>,
    _ __IA: vDSP_Stride,
    _ __B: UnsafePointer<Float>,
    _ __C: UnsafeMutablePointer<Float>,
    _ __IC: vDSP_Stride,
    _ __N: vDSP_Length
) {
    var i = 0
    while i < __N {
        __C[i * __IC] = __A[i * __IA] * __B.pointee
        i += 1
    }
}

/// Single-precision add scalar: C = A + (*B).
@inlinable
public func vDSP_vsadd(
    _ __A: UnsafePointer<Float>,
    _ __IA: vDSP_Stride,
    _ __B: UnsafePointer<Float>,
    _ __C: UnsafeMutablePointer<Float>,
    _ __IC: vDSP_Stride,
    _ __N: vDSP_Length
) {
    var i = 0
    while i < __N {
        __C[i * __IC] = __A[i * __IA] + __B.pointee
        i += 1
    }
}

/// Single-precision lower threshold (clip up): C = max(A, *B). With *B == 0 this is ReLU.
@inlinable
public func vDSP_vthres(
    _ __A: UnsafePointer<Float>,
    _ __IA: vDSP_Stride,
    _ __B: UnsafePointer<Float>,
    _ __C: UnsafeMutablePointer<Float>,
    _ __IC: vDSP_Stride,
    _ __N: vDSP_Length
) {
    let threshold = __B.pointee
    var i = 0
    while i < __N {
        let v = __A[i * __IA]
        __C[i * __IC] = v < threshold ? threshold : v
        i += 1
    }
}

/// Single-precision scalar multiply and add: D = A * (*B) + C.
@inlinable
public func vDSP_vsma(
    _ __A: UnsafePointer<Float>,
    _ __IA: vDSP_Stride,
    _ __B: UnsafePointer<Float>,
    _ __C: UnsafePointer<Float>,
    _ __IC: vDSP_Stride,
    _ __D: UnsafeMutablePointer<Float>,
    _ __ID: vDSP_Stride,
    _ __N: vDSP_Length
) {
    let scalar = __B.pointee
    var i = 0
    while i < __N {
        __D[i * __ID] = __A[i * __IA] * scalar + __C[i * __IC]
        i += 1
    }
}

/// Single-precision sum of vector elements: *C = sum(A).
@inlinable
public func vDSP_sve(
    _ __A: UnsafePointer<Float>,
    _ __IA: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Float>,
    _ __N: vDSP_Length
) {
    var sum: Float = 0
    var i = 0
    while i < __N {
        sum += __A[i * __IA]
        i += 1
    }
    __C.pointee = sum
}

/// Single-precision element-wise exponential: y = exp(x).
@inlinable
public func vvexpf(
    _ y: UnsafeMutablePointer<Float>,
    _ x: UnsafePointer<Float>,
    _ n: UnsafePointer<Int32>
) {
    var i = 0
    while i < n.pointee {
        y[i] = expf(x[i])
        i += 1
    }
}

/// Single-precision element-wise square root: y = sqrt(x).
@inlinable
public func vvsqrtf(
    _ y: UnsafeMutablePointer<Float>,
    _ x: UnsafePointer<Float>,
    _ n: UnsafePointer<Int32>
) {
    var i = 0
    while i < n.pointee {
        y[i] = sqrtf(x[i])
        i += 1
    }
}

/// Single-precision element-wise reciprocal: y = 1 / x.
@inlinable
public func vvrecf(
    _ y: UnsafeMutablePointer<Float>,
    _ x: UnsafePointer<Float>,
    _ n: UnsafePointer<Int32>
) {
    var i = 0
    while i < n.pointee {
        y[i] = 1 / x[i]
        i += 1
    }
}
#endif  // canImport(Accelerate)
