using LinearAlgebra, SparseArrays, LDLFactorizations

nx = 2
nu = 1
T = 3
m = nx + nu
n = (T + 1) * m
n_eq = (T + 1) * nx

FloatType = Float64
A = [1.0 0.1; 0.0 1.0]
B = [0.0; 0.1;;]

M1 = [sparse(I, nx, nx) spzeros(FloatType, nx, nu)]
M2 = [A B]

M = spzeros(FloatType, n_eq, n)
M[1:nx, 1:m] = M1
for t in 1:T
    rs = t*nx + 1;  re = (t+1)*nx
    M[rs:re, t*m+1       : (t+1)*m] =  M1
    M[rs:re, (t-1)*m+1   : t*m    ] = -M2
end

K = [sparse(I, n, n) M'; M spzeros(FloatType, n_eq, n_eq)]
println("K structurally symmetric? ", issymmetric(K))

F = ldl(K)
println("LDL factorization successful.")

# test KKT solve
g = zeros(FloatType, n_eq)
g[1:nx] .= [1.2, 3.4]
RHS = zeros(FloatType, n + n_eq)
buffer1 = randn(n)

RHS[1:n] .= buffer1
RHS[n+1:n+n_eq] .= g

sol = F \ RHS
v = sol[1:n]
println("v length: ", length(v))

# check constraint M*v = g
println("max |M v - g| : ", maximum(abs.(M*v - g)))

