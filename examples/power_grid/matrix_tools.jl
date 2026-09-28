@inline function add_bias!(mat, bias)
    cols = size(mat, 2)
    @inbounds @simd for j in 1:cols
        @views mat[:, j] .+= bias
    end
    return mat
end

@inline function activation_sigma!(activ, sigma_buf, preactiv)
    @static if Sys.isapple() && isdefined(@__MODULE__, :AppleAccelerate)
        @. sigma_buf = -abs(preactiv)
        AppleAccelerate.exp!(activ, sigma_buf)
        AppleAccelerate.log1p!(sigma_buf, activ)
        @inbounds @simd for i in eachindex(activ)
            val, e, logarithm = preactiv[i], activ[i], sigma_buf[i]
            activ[i] = max(val, zero(val)) + logarithm
            sigma_buf[i] = val >= 0 ? inv(1 + e) : e / (1 + e)
        end
        return activ
    end
    @inbounds @simd for i in eachindex(activ)
        val          = preactiv[i]
        e = exp(-abs(val))
        activ[i] = max(val, zero(val)) + log1p(e)
        sigma_buf[i] = val >= 0 ? inv(1 + e) : e / (1 + e)
    end
    return activ
end

@inline function activation_sigma_only!(sigma_buf, preactiv, scratch)
    @static if Sys.isapple() && isdefined(@__MODULE__, :AppleAccelerate)
        @. scratch = -abs(preactiv)
        AppleAccelerate.exp!(sigma_buf, scratch)
        @inbounds @simd for i in eachindex(sigma_buf)
            e = sigma_buf[i]
            sigma_buf[i] = preactiv[i] >= 0 ? inv(1 + e) : e / (1 + e)
        end
    else
        sigma_buf .= NNlib.σ.(preactiv)
    end
    return sigma_buf
end
# precompile(activation_sigma!, (Function, MMatrix{10,10,Float64,100}, MMatrix{10,10,Float64,100}))


@inline function hadamard!(dest, rhs)
    @inbounds @simd for i in eachindex(dest)
        dest[i] *= rhs[i]
    end
    return dest
end
# precompile(hadamard!, (MMatrix{10,10,Float64,100}, MMatrix{10,10,Float64,100}))



@inline function mmul_add_matrix!(dest, src1, src2)
    return mul!(dest, src1, src2, one(eltype(dest)), one(eltype(dest)))
end
# precompile(mul_add_matrix!, (MMatrix{10,10,Float64,100}, MMatrix{10,10,Float64,100}, MMatrix{10,10,Float64,100}))
