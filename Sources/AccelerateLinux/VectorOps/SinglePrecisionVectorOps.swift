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

// MARK: - Additional single-precision vDSP shims
//
// Further single-precision (Float) counterparts used by AccelerateLinux
// consumers (ValueStore, QueryDecomposer, SwiftEpisteme, UniversalValueSystem).
// Behavior mirrors the documented Accelerate semantics; these scalar loops are
// not SIMD-optimized.

/// Single-precision dot product: *C = sum(A * B).
@inlinable
public func vDSP_dotpr(
    _ __A: UnsafePointer<Float>,
    _ __IA: vDSP_Stride,
    _ __B: UnsafePointer<Float>,
    _ __IB: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Float>,
    _ __N: vDSP_Length
) {
    var sum: Float = 0
    for i in 0..<Int(__N) {
        sum += __A[i * __IA] * __B[i * __IB]
    }
    __C.pointee = sum
}

/// Single-precision maximum value: *C = max(A).
@inlinable
public func vDSP_maxv(
    _ __A: UnsafePointer<Float>,
    _ __IA: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Float>,
    _ __N: vDSP_Length
) {
    let n = Int(__N)
    guard n > 0 else { __C.pointee = 0; return }
    var m = __A[0]
    for i in 1..<n { m = Swift.max(m, __A[i * __IA]) }
    __C.pointee = m
}

/// Single-precision minimum value: *C = min(A).
@inlinable
public func vDSP_minv(
    _ __A: UnsafePointer<Float>,
    _ __IA: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Float>,
    _ __N: vDSP_Length
) {
    let n = Int(__N)
    guard n > 0 else { __C.pointee = 0; return }
    var m = __A[0]
    for i in 1..<n { m = Swift.min(m, __A[i * __IA]) }
    __C.pointee = m
}

/// Single-precision maximum magnitude: *C = max(|A|).
@inlinable
public func vDSP_maxmgv(
    _ __A: UnsafePointer<Float>,
    _ __IA: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Float>,
    _ __N: vDSP_Length
) {
    let n = Int(__N)
    guard n > 0 else { __C.pointee = 0; return }
    var m = Swift.abs(__A[0])
    for i in 1..<n { m = Swift.max(m, Swift.abs(__A[i * __IA])) }
    __C.pointee = m
}

/// Single-precision maximum value and its index: *C = max, *I = index (× stride).
@inlinable
public func vDSP_maxvi(
    _ __A: UnsafePointer<Float>,
    _ __IA: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Float>,
    _ __I: UnsafeMutablePointer<vDSP_Length>,
    _ __N: vDSP_Length
) {
    let n = Int(__N)
    guard n > 0 else { __C.pointee = 0; __I.pointee = 0; return }
    var m = __A[0]
    var best = 0
    for i in 1..<n {
        let v = __A[i * __IA]
        if v > m { m = v; best = i }
    }
    __C.pointee = m
    __I.pointee = vDSP_Length(best * __IA)
}

/// Single-precision minimum value and its index: *C = min, *I = index (× stride).
@inlinable
public func vDSP_minvi(
    _ __A: UnsafePointer<Float>,
    _ __IA: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Float>,
    _ __I: UnsafeMutablePointer<vDSP_Length>,
    _ __N: vDSP_Length
) {
    let n = Int(__N)
    guard n > 0 else { __C.pointee = 0; __I.pointee = 0; return }
    var m = __A[0]
    var best = 0
    for i in 1..<n {
        let v = __A[i * __IA]
        if v < m { m = v; best = i }
    }
    __C.pointee = m
    __I.pointee = vDSP_Length(best * __IA)
}

/// Single-precision mean value: *C = sum(A) / N.
@inlinable
public func vDSP_meanv(
    _ __A: UnsafePointer<Float>,
    _ __IA: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Float>,
    _ __N: vDSP_Length
) {
    let n = Int(__N)
    guard n > 0 else { __C.pointee = 0; return }
    var sum: Float = 0
    for i in 0..<n { sum += __A[i * __IA] }
    __C.pointee = sum / Float(n)
}

/// Single-precision mean of squares: *C = sum(A * A) / N.
@inlinable
public func vDSP_measqv(
    _ __A: UnsafePointer<Float>,
    _ __IA: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Float>,
    _ __N: vDSP_Length
) {
    let n = Int(__N)
    guard n > 0 else { __C.pointee = 0; return }
    var sum: Float = 0
    for i in 0..<n { let v = __A[i * __IA]; sum += v * v }
    __C.pointee = sum / Float(n)
}

/// Single-precision root mean square: *C = sqrt(sum(A * A) / N).
@inlinable
public func vDSP_rmsqv(
    _ __A: UnsafePointer<Float>,
    _ __IA: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Float>,
    _ __N: vDSP_Length
) {
    let n = Int(__N)
    guard n > 0 else { __C.pointee = 0; return }
    var sum: Float = 0
    for i in 0..<n { let v = __A[i * __IA]; sum += v * v }
    __C.pointee = sqrtf(sum / Float(n))
}

/// Single-precision sum of squares: *C = sum(A * A).
@inlinable
public func vDSP_svesq(
    _ __A: UnsafePointer<Float>,
    _ __IA: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Float>,
    _ __N: vDSP_Length
) {
    var sum: Float = 0
    for i in 0..<Int(__N) { let v = __A[i * __IA]; sum += v * v }
    __C.pointee = sum
}

/// Single-precision scalar divided by vector: C = (*A) / B.
@inlinable
public func vDSP_svdiv(
    _ __A: UnsafePointer<Float>,
    _ __B: UnsafePointer<Float>,
    _ __IB: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Float>,
    _ __IC: vDSP_Stride,
    _ __N: vDSP_Length
) {
    let a = __A.pointee
    for i in 0..<Int(__N) { __C[i * __IC] = a / __B[i * __IB] }
}

/// Single-precision clip to a range: D = min(max(A, *B), *C).
@inlinable
public func vDSP_vclip(
    _ __A: UnsafePointer<Float>,
    _ __IA: vDSP_Stride,
    _ __B: UnsafePointer<Float>,
    _ __C: UnsafePointer<Float>,
    _ __D: UnsafeMutablePointer<Float>,
    _ __ID: vDSP_Stride,
    _ __N: vDSP_Length
) {
    let lo = __B.pointee
    let hi = __C.pointee
    for i in 0..<Int(__N) {
        let v = __A[i * __IA]
        __D[i * __ID] = v < lo ? lo : (v > hi ? hi : v)
    }
}

/// Single-precision clear (zero-fill): C = 0.
@inlinable
public func vDSP_vclr(
    _ __C: UnsafeMutablePointer<Float>,
    _ __IC: vDSP_Stride,
    _ __N: vDSP_Length
) {
    for i in 0..<Int(__N) { __C[i * __IC] = 0 }
}

/// Single-precision fill with a scalar: C = (*A).
@inlinable
public func vDSP_vfill(
    _ __A: UnsafePointer<Float>,
    _ __C: UnsafeMutablePointer<Float>,
    _ __IC: vDSP_Stride,
    _ __N: vDSP_Length
) {
    let a = __A.pointee
    for i in 0..<Int(__N) { __C[i * __IC] = a }
}

/// Single-precision vector multiply and add: D = A * B + C.
@inlinable
public func vDSP_vma(
    _ __A: UnsafePointer<Float>,
    _ __IA: vDSP_Stride,
    _ __B: UnsafePointer<Float>,
    _ __IB: vDSP_Stride,
    _ __C: UnsafePointer<Float>,
    _ __IC: vDSP_Stride,
    _ __D: UnsafeMutablePointer<Float>,
    _ __ID: vDSP_Stride,
    _ __N: vDSP_Length
) {
    for i in 0..<Int(__N) {
        __D[i * __ID] = __A[i * __IA] * __B[i * __IB] + __C[i * __IC]
    }
}

/// Single-precision negation: C = -A.
@inlinable
public func vDSP_vneg(
    _ __A: UnsafePointer<Float>,
    _ __IA: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Float>,
    _ __IC: vDSP_Stride,
    _ __N: vDSP_Length
) {
    for i in 0..<Int(__N) { __C[i * __IC] = -__A[i * __IA] }
}

/// Single-precision element-wise square: C = A * A.
@inlinable
public func vDSP_vsq(
    _ __A: UnsafePointer<Float>,
    _ __IA: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Float>,
    _ __IC: vDSP_Stride,
    _ __N: vDSP_Length
) {
    for i in 0..<Int(__N) { let v = __A[i * __IA]; __C[i * __IC] = v * v }
}

/// Single-precision matrix transpose. Mirrors `vDSP_mtransD`.
@inlinable
public func vDSP_mtrans(
    _ __A: UnsafePointer<Float>,
    _ __IA: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Float>,
    _ __IC: vDSP_Stride,
    _ __M: vDSP_Length,
    _ __N: vDSP_Length
) {
    var __A = __A
    for i in 0..<__N {
        var __C = __C.advanced(by: Int(i) * __IC)
        for _ in 0..<__M {
            __C.pointee = __A.pointee
            __A = __A.advanced(by: __IA)
            __C = __C.advanced(by: Int(__N) * __IC)
        }
    }
}

/// Single-precision matrix multiply: C = A(M×P) * B(P×N), overwriting C(M×N).
@inlinable
public func vDSP_mmul(
    _ __A: UnsafePointer<Float>,
    _ __IA: vDSP_Stride,
    _ __B: UnsafePointer<Float>,
    _ __IB: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Float>,
    _ __IC: vDSP_Stride,
    _ __M: vDSP_Length,
    _ __N: vDSP_Length,
    _ __P: vDSP_Length
) {
    let M = Int(__M), N = Int(__N), P = Int(__P)
    for i in 0..<M {
        for j in 0..<N {
            var sum: Float = 0
            for k in 0..<P {
                sum += __A[(i * P + k) * __IA] * __B[(k * N + j) * __IB]
            }
            __C[(i * N + j) * __IC] = sum
        }
    }
}

/// Single-precision matrix move (copy an M-column × N-row submatrix).
@inlinable
public func vDSP_mmov(
    _ __A: UnsafePointer<Float>,
    _ __C: UnsafeMutablePointer<Float>,
    _ __M: vDSP_Length,
    _ __N: vDSP_Length,
    _ __TA: vDSP_Length,
    _ __TC: vDSP_Length
) {
    let cols = Int(__M), rows = Int(__N), ta = Int(__TA), tc = Int(__TC)
    for r in 0..<rows {
        for c in 0..<cols {
            __C[r * tc + c] = __A[r * ta + c]
        }
    }
}

// MARK: - Additional single-precision vForce shims

/// Single-precision element-wise cosine: y = cos(x).
@inlinable
public func vvcosf(
    _ y: UnsafeMutablePointer<Float>,
    _ x: UnsafePointer<Float>,
    _ n: UnsafePointer<Int32>
) {
    var i = 0
    while i < n.pointee { y[i] = cosf(x[i]); i += 1 }
}

/// Single-precision element-wise sine: y = sin(x).
@inlinable
public func vvsinf(
    _ y: UnsafeMutablePointer<Float>,
    _ x: UnsafePointer<Float>,
    _ n: UnsafePointer<Int32>
) {
    var i = 0
    while i < n.pointee { y[i] = sinf(x[i]); i += 1 }
}

/// Single-precision element-wise tangent: y = tan(x).
@inlinable
public func vvtanf(
    _ y: UnsafeMutablePointer<Float>,
    _ x: UnsafePointer<Float>,
    _ n: UnsafePointer<Int32>
) {
    var i = 0
    while i < n.pointee { y[i] = tanf(x[i]); i += 1 }
}

/// Single-precision element-wise hyperbolic tangent: y = tanh(x).
@inlinable
public func vvtanhf(
    _ y: UnsafeMutablePointer<Float>,
    _ x: UnsafePointer<Float>,
    _ n: UnsafePointer<Int32>
) {
    var i = 0
    while i < n.pointee { y[i] = tanhf(x[i]); i += 1 }
}

/// Single-precision element-wise natural logarithm: y = log(x).
@inlinable
public func vvlogf(
    _ y: UnsafeMutablePointer<Float>,
    _ x: UnsafePointer<Float>,
    _ n: UnsafePointer<Int32>
) {
    var i = 0
    while i < n.pointee { y[i] = logf(x[i]); i += 1 }
}

/// Single-precision element-wise power: z = x ^ y (x is the base, y the exponent).
@inlinable
public func vvpowf(
    _ z: UnsafeMutablePointer<Float>,
    _ y: UnsafePointer<Float>,
    _ x: UnsafePointer<Float>,
    _ n: UnsafePointer<Int32>
) {
    var i = 0
    while i < n.pointee { z[i] = powf(x[i], y[i]); i += 1 }
}
#endif  // canImport(Accelerate)
