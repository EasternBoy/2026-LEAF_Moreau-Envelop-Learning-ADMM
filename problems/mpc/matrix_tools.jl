@inbounds function add_bias!(mat::VecOrMat{FloatType}, bias)
    cols = size(mat, 2)
    @inbounds @simd for j in 1:cols
        @views mat[:, j] .+= bias
    end
    return mat
end

@inbounds function activation_sigma!(activ::VecOrMat{FloatType}, sigma_buf::VecOrMat{FloatType}, preactiv)
    @simd for i in eachindex(activ)
        val          = preactiv[i]
        activ[i]     = NNlib.softplus(val)
        sigma_buf[i] = NNlib.σ(val)
    end
    return activ
end
# precompile(activation_sigma!, (Function, MMatrix{10,10,Float64,100}, MMatrix{10,10,Float64,100}))


@inbounds function hadamard!(dest::VecOrMat{FloatType}, rhs::VecOrMat{FloatType})
    @simd for i in eachindex(dest)
        dest[i] *= rhs[i]
    end
    return dest
end

@inline @inbounds function mul_add!(dest, m1, m2)
    buff = copy(dest)
    mul!(buff, m1, m2)
    dest .+= buff
    return dest
end 



