using JuMP

if !(@isdefined(FloatType))
    const FloatType = Float64
end

#========== logsum  ========#
struct logsum
    w::FloatType
end

function (obj::logsum)(x, u, model = nothing)
    return log(sum(exp.(x))) + obj.w*dot(u, u)
end

#========== quadratic  ========#
struct quadratic
    Q::Union{FloatType, VecOrMat{FloatType}}
    R::Union{FloatType, VecOrMat{FloatType}}
end

function (obj::quadratic)(x, u, model = nothing)
    Q  = obj.Q
    R  = obj.R
    return dot(x,Q,x) + dot(u,R,u)
end

#========== L1norm  ========#
struct L1norm
    nx::Int
    nu::Int
    Q::VecOrMat{FloatType}
    R::VecOrMat{FloatType}
end

function (obj::L1norm)(x, u, model = nothing)
    nx = obj.nx
    nu = obj.nu
    Q = obj.Q
    R = obj.R

    if model !== nothing
        tx = @variable(model, [1:nx])     # epigraph for |Q*x_k|
        tu = @variable(model, [1:nu])     # epigraph for |R*u_k|

        @constraint(model,  Q * x .<= tx)
        @constraint(model, -Q * x .<= tx)

        @constraint(model,  R * u .<= tu)
        @constraint(model, -R * u .<= tu)

        @constraint(model,  0 .<= tx)
        @constraint(model,  0 .<= tx)

        return sum(tx) + sum(tu)
    else
        return sum(abs.(Q*x)) + sum(abs.(R*u))
    end
end

#========== power_share  ========#
struct power_share
    n::Int
    a::VecOrMat{FloatType}
    η1::FloatType
    η2::FloatType
end

function (obj::power_share)(x, rl, model = nothing)
    a = obj.a
    n = obj.n
    if model !== nothing
        d = @variable(model, [1:n]) 
        for i in 1:n
            @constraint(model,  d[i] >= 1/x[i] - 1/a[i])
            @constraint(model,  d[i] >= 0)
        end
        return sum(d) + obj.η1 * dot(rl, rl) + obj.η2 * dot((x - a)./a, (x - a)./a)
    else
        return sum(1/min(x[i], a[i]) - 1/a[i] for i in 1:n) + obj.η1 * dot(rl, rl) + obj.η2 * dot((x - a)./a, (x - a)./a)
    end
end