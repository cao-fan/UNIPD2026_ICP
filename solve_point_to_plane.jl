module SoftICP

export register

using CUDA
using StaticArrays
using LinearAlgebra
using Random
using Rotations
Random.seed!(123)

include("point_cloud_utils.jl")
using .PointCloudUtils
include("fixed_centroids_gmm.jl")
using. FixedCentroidsGMM

################ GEOMETRIC TRANSFORMATIONS
function get_rot_vec(R)
    return CUDA.@allowscalar Rotations.params(RotationVec(R))
end

function hat(v::AbstractVector{T}) where T
    #  0 -z  y
    #  z  0 -x 
    # -y  x  0
    # Column major
    x = v[1]
    y = v[2]
    z = v[3]
    return SMatrix{3,3,T,9}(0,z,-y, -z,0,x, y,-x,0)
end

function rodriguez(delta::AbstractVector{T}) where T
    delta = SVector{3,T}(delta)
    delta_norm = norm(delta)
    if isapprox(delta_norm,zero(delta_norm),atol=1e-8)
        return SMatrix{3,3,T}(I)
    end
    normalized_delta = delta/delta_norm
    S = hat(normalized_delta)
    S_2 = normalized_delta*transpose(normalized_delta) - I
    R = I + sin(delta_norm)*S + (1-cos(delta_norm))*S_2
    return R
end

function left_jacobian(theta::AbstractVector{T}) where T
    angle = norm(theta)
    theta_hat = hat(theta)
    if isapprox(angle,zero(angle))
        return SMatrix{3,3,T}(I)#I(3) + theta_hat/2
    end
    a = (1-cos(angle))/(angle^2) # angle -> 0 => a -> 1/2
    b = (angle - sin(angle))/(angle^3) # angle -> 0 => b theta_hat^2 -> 0
    return I(3) + a * theta_hat + b * theta_hat^2
end

function left_jacobian_inverse(theta::AbstractVector{T}) where T
    angle = norm(theta)
    theta_hat = hat(theta)
    if isapprox(angle,zero(angle))
        return SMatrix{3,3,T}(I)#I(3) - theta_hat/2
    end
    a = -1/2
    b = 1/angle^2 - (1+cos(angle))/(2*angle*sin(angle))
    return I(3) + a * theta_hat + b * theta_hat^2
end

############# PDFs

function uniform_pdf(T::AbstractArray)
    return inv(prod(maximum(T;dims=2) - minimum(T;dims=2)))
end

function normal_pdf(x,mu,C)
    F = cholesky(Symmetric(C))
    c = inv(prod(sqrt(2*pi).*diag(F.U)))
    d = x - mu
    return c * exp.(-dot(d, F \ d)/2)
end

############ POSTERIOR

function populate_P!(P,R_cu,t_cu,S,T,source_covs,target_covs, gaussian_priors, uniform_dist_prob, outlier_prior = 0.1f0, covariance_scaling=1f0)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    M = size(source_covs,3)
    N = size(target_covs,3)

    indices = index:stride:length(P)
    
    CI = CartesianIndices(P)

    
    R = SMatrix{3,3}(R_cu)
    t = SVector{3}(t_cu)

    for k in indices
        pair = CI[k]
        j = pair[1]
        i = pair[2]

        #R*Cx*Rt + Cy
        if j <= M
            Cx = SMatrix{3,3}(@view(source_covs[:,:,j]))
            Cy = SMatrix{3,3}(@view(target_covs[:,:,i]))
            x = SVector{3}(view(S,:,j))
            y = SVector{3}(view(T,:,i))

            C = covariance_scaling * Symmetric(R * Cx * transpose(R) + Cy)
            μ = SVector{3}(R * x + t)
            P[j,i] = normal_pdf(y,μ,C) * gaussian_priors[j]
        else
            P[j,i] = uniform_dist_prob * outlier_prior
        end
    end
    return nothing
end

function compute_posteriors_kernel!(P, S, T, source_covs, target_covs, R, t, source_mixing_prior, outlier_prior_prob, covariance_scaling = 1f0)
    uniform_dist_prob = uniform_pdf(T)

    kernel = @cuda launch=false populate_P!(P,R,t,S,T,source_covs,target_covs, source_mixing_prior, uniform_dist_prob, outlier_prior_prob, covariance_scaling)
    config = launch_configuration(kernel.fun)
    threads = min(length(P), config.threads)
    blocks = cld(length(P), threads)
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks populate_P!(P,R,t,S,T,source_covs,target_covs, source_mixing_prior, uniform_dist_prob, outlier_prior_prob, covariance_scaling)
    end
    P_row_sum = sum(P;dims=1)
    if isapprox(outlier_prior_prob,zero(outlier_prior_prob))
        P_row_sum .= P_row_sum .+ eps(Float32)
    end
    P .= abs.(P ./ P_row_sum)
    return nothing
end

####################### Linear approximation nonlinear least squares

function prepare_A_b!(A,b,_R,S,T,S_covs,T_covs,P)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    M = size(S_covs,3)
    N = size(T_covs,3)

    indices = index:stride:(N)
    
    CI = CartesianIndices((N,)) 

    local_A = @SMatrix zeros(Float32,6,6)
    local_b = @SVector zeros(Float32,6)

    R = SMatrix{3,3}(_R)
    R_t = transpose(R)
    for k in indices
        pair = CI[k]
        i = pair[1]

        Cy = SMatrix{3,3}(view(T_covs,:,:,i))
        y = SVector{3}(view(T,:,i))
        for j in axes(S_covs,3) #1:M
            Cx = SMatrix{3,3}(view(S_covs,:,:,j))
            x = SVector{3}(view(S,:,j))
            M_mat = Symmetric(R * Cx * R_t + Cy)
            F = cholesky(M_mat)
            F_inv = SMatrix{3,3}(inv(F))
            R_x = R * x
            d = R_x - y
            p = F_inv * d 
            R_x_hat = SMatrix{3,3}(hat(R_x))
            Q = hcat(R_x_hat,-SMatrix{3,3}(I))
            Q_t = transpose(Q)

            b_ji = Q_t * p
            A_ji = Q_t * F_inv * Q
            local_b += P[j,i] * b_ji / N #P[j,i] * b_ji
            local_A += P[j,i] * A_ji / N #P[j,i] * A_ji
        end
    end

    for idx in eachindex(b)
        CUDA.@atomic b[idx] = b[idx] + local_b[idx]
    end
    for idx in eachindex(A)
        CUDA.@atomic A[idx] = A[idx] + local_A[idx]
    end
    return nothing
end
function solve_A_b_for_theta_kernel(A,b,R,t,S,T,S_covs,T_covs,P;prior_R,prior_t,Sigma_inv_R,Sigma_inv_t)
    M = size(S_covs,3)
    N = size(T_covs,3)
    
    kernel = @cuda launch=false prepare_A_b!(A,b,R,S,T,S_covs,T_covs,P)
    config = launch_configuration(kernel.fun)
    threads = min(N, config.threads)
    blocks = cld(N, threads)
    
    A .= 0f0
    b .= 0f0
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks prepare_A_b!(A,b,R,S,T,S_covs,T_covs,P)
    end
    CUDA.@allowscalar begin
        τ = get_rot_vec(R*transpose(prior_R)) |> Array |> cu
        Jₗ⁻¹ = left_jacobian_inverse(τ) |> Array |> cu
        Jₗ⁻ᵀ = transpose(Jₗ⁻¹)
        A[1:3,1:3] .+= Jₗ⁻ᵀ * Sigma_inv_R * Jₗ⁻¹ # * N
        A[4:6,4:6] .+= Sigma_inv_t # * N
        b[1:3] .+= -Jₗ⁻ᵀ * Sigma_inv_R * τ # * N
        b[4:6] .+= Sigma_inv_t * prior_t # * N
    end
    θ = A \ b
    # Subtract t_0 to get Δθ instead of θ
    Δθ = θ
    Δθ[4:6] = θ[4:6] - t
    return Δθ
end


function p2p_ml_cost_kernel(du,R,t,p)
    S, T, S_covs, T_covs, P = p
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    M = size(S_covs,3)
    N = size(T_covs,3)

    indices = index:stride:(M*N)
    
    CI = CartesianIndices((M,N))
    
    R = SMatrix{3,3}(R)
    t = SVector{3}(t)

    for k in indices
        pair = CI[k]
        j = pair[1]
        i = pair[2]

        Cx = SMatrix{3,3}(@view(S_covs[:,:,j]))
        Cy = SMatrix{3,3}(@view(T_covs[:,:,i]))
        x = SVector{3}(view(S,:,j))
        y = SVector{3}(view(T,:,i))
        W = Symmetric(R * Cx * transpose(R) + Cy)
        d = y - (R * x + t)
        F = cholesky(W)
        du[j,i] = P[j,i]/N * (dot(d, (F \ d)) + logdet(F))
    end
    return nothing 
end

function compute_p2p_ml_cost_kernel_with_prior(gpu_R, gpu_t, gpu_p, prior_R,prior_t,Sigma_inv_R,Sigma_inv_t)
    S, T, S_covs, T_covs, P, cache= gpu_p
    M = size(S_covs,3)
    N = size(T_covs,3)
    _gpu_p = (S, T, S_covs, T_covs, P)
    kernel = @cuda launch=false p2p_ml_cost_kernel(cache,gpu_R,gpu_t,_gpu_p)
    config = launch_configuration(kernel.fun)
    threads = min(M*N, config.threads)
    blocks = cld(M*N, threads)
    
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks p2p_ml_cost_kernel(cache,gpu_R,gpu_t,_gpu_p)
    end
    d = CUDA.@allowscalar cu(Array(get_rot_vec(gpu_R*transpose(prior_R))))
    return sum(cache) + transpose(gpu_t-prior_t)*Sigma_inv_t*(gpu_t-prior_t) + transpose(d)*Sigma_inv_R*d
end
#################

function icp_plane2plane_ml_approximation_closed_form(source_pc::AbstractMatrix{T}, target_pc::AbstractMatrix{T};prior_R=cu(I(3)), prior_t=CUDA.zeros(3), Sigma_inv_R=CUDA.zeros(3,3), Sigma_inv_t=CUDA.zeros(3,3), outlier_prior_prob=0.0, max_iters = 100, max_population=100, gauss_newton_max_iters=10, grad_eps_termination=100*eps(Float32)) where T 
    function update_R(theta,R)
        CUDA.@allowscalar begin
            return rodriguez(theta[1:3]) * R
        end
    end
    function get_XYZ_jacobian(R) 
        CUDA.@allowscalar begin 
            xyz = Rotations.params(RotXYZ(R)) 
            Rze2 = RotXYZ(0,0,xyz[3])[:,2] 
            RzRye1 = RotXYZ(0,0,xyz[3]) * RotXYZ(0,xyz[2],0)[:,1] 
            J = [RzRye1;;Rze2;;[1f0,0,0]] 
            return J 
        end 
    end 
    function get_Sigma_inv_XYZ(Sigma_inv,R) 
        J = pinv(cu(Matrix(get_XYZ_jacobian(R)))) 
        return transpose(J) * Sigma_inv * J 
    end
    Random.seed!(42)
    # Subsample the point cloud
    resampled_source_pc, resampled_source_covs, source_mixing_prior = hard_predict_gmm_kernel(cu(source_pc), max_population)#hard_predict_gmm(source_pc[:,idx_subsample], max_population) .|> cu
    resampled_target_pc, resampled_target_covs, target_mixing_prior = hard_predict_gmm_kernel(cu(target_pc), max_population)#hard_predict_gmm(target_pc[:,idx_subsample], max_population) .|> cu
    
    M, N = size(resampled_source_pc,2),size(resampled_target_pc,2)

    # Initialize the parameters
    R = cu(Matrix(prior_R))
    t = cu(Vector(prior_t))
    # Inizialize the optimization parameter
    θ = cu(zeros(Float32,6))
    θ[1:3] = get_rot_vec(R) #.+ 1e-3*CUDA.randn(3)
    θ[4:end] = t #.+ 1e-3*CUDA.randn(3)
    # Initialize caches
    P = CUDA.zeros(Float32,M+1,N) # Posterior cache
    du_cache = CUDA.zeros(Float32,M,N) # Cost function cache
    # Linear solve cache
    A_cache = cu(zeros(6,6))
    b_cache = cu(zeros(6,))
    # Temp variables
    old_theta = similar(θ)
    temp_theta = similar(θ)
    grad_cache = similar(θ)

    STEP_SIZE = 1.0f0
    
    # Initialize optimization variables
    loss = Inf
    # Optimization loop
    for i in 1:max_iters
        # E-step
        covariance_scaling = 1.f0
        compute_posteriors_kernel!(P, resampled_source_pc, resampled_target_pc, resampled_source_covs, resampled_target_covs, R, t, source_mixing_prior, outlier_prior_prob, covariance_scaling)
        # M-step
        p = (resampled_source_pc, resampled_target_pc, resampled_source_covs, resampled_target_covs, P, du_cache)
        _Sigma_inv_R = get_Sigma_inv_XYZ(Sigma_inv_R,R)
        initial_loss = compute_p2p_ml_cost_kernel_with_prior(R,t,p,prior_R, prior_t, _Sigma_inv_R, Sigma_inv_t)
        # Gauss-Newton algorithm
        old_theta .= θ
        # @debug "Old theta" old_theta initial_loss
        for inner_iter in 1:gauss_newton_max_iters
            _Sigma_inv_R = get_Sigma_inv_XYZ(Sigma_inv_R,R)
            # Gauss-Newton step
            Δθ = solve_A_b_for_theta_kernel(
                A_cache,b_cache,R,t,resampled_source_pc,resampled_target_pc,resampled_source_covs,resampled_target_covs,P;
                prior_R=prior_R, prior_t=prior_t, Sigma_inv_R=_Sigma_inv_R, Sigma_inv_t=Sigma_inv_t
            )
            # @debug "" Δθ
            step_size = STEP_SIZE
            new_R = update_R(step_size*Δθ,R)
            new_t = t + step_size*Δθ[4:6]
            # If cost is not decreasing anymore stop
            temp_theta[1:3] = get_rot_vec(new_R)
            temp_theta[4:6] = new_t
            _Sigma_inv_R = get_Sigma_inv_XYZ(Sigma_inv_R,new_R)
            new_loss = compute_p2p_ml_cost_kernel_with_prior(new_R,new_t,p,prior_R, prior_t, _Sigma_inv_R, Sigma_inv_t)
            # @debug "" temp_theta norm(temp_theta - old_theta) new_loss
            if new_loss < initial_loss
                R = new_R
                t = new_t
                θ = temp_theta
                initial_loss = new_loss
                if inner_iter==gauss_newton_max_iters
                    loss = new_loss
                end
            else
                # @debug "Breaking at iter $inner_iter"
                loss = new_loss
                break
            end
        end
        
        @info "Iteration $i/$max_iters" loss
        
        # Termination criteria: stop when R and t do not change by much
        norm_theta_change = norm(θ - old_theta)
        # @debug "" norm_theta_change
        if norm_theta_change < grad_eps_termination
            break
        end
    end
    plot = plot_pc(target_pc;markersize=1.5)
    transformed_source = Array(R) * source_pc .+ Array(t)
    plot_pc!(transformed_source;markersize=1.5)
    return Array(R), Array(t)
end


function register(source, target;outlier_prior_prob=0.0, R_hint=Matrix(I(3)), t_hint=zeros(3), sigmas_R_rad=zeros(3), sigmas_t=zeros(3), max_iters=100, max_population=1000)
    # Normalize the point clouds
    source_mean,target_mean,scaling = normalize_point_clouds(source, target)
    normalized_source = scaling*(source .- source_mean)
    normalized_target = scaling*(target .- target_mean)
    # Initialize the priors
    # R' = R_hint
    rescaled_R_hint = Matrix(R_hint)
    # t' = γ(R_hint*source_mean - target_mean + t_hint)
    rescaled_t_hint = Vector(scaling*(R_hint*source_mean .- target_mean .+ t_hint))
    Sigma_R = if sigmas_R_rad isa AbstractVector
        diagm(sigmas_R_rad.^2)
    elseif sigmas_R_rad isa AbstractMatrix
        sigmas_R_rad
    else
        zeros(3,3)
    end
    Sigma_inv_R = pinv(Sigma_R)
    Sigma_t = if sigmas_t isa AbstractVector
        diagm((scaling.*sigmas_t).^2)
    elseif sigmas_t isa AbstractMatrix
        sigmas_t
    else
        zeros(3,3)
    end
    Sigma_inv_t = pinv(Sigma_t)
    opt_R, opt_t = icp_plane2plane_ml_approximation_closed_form(
        normalized_source,
        normalized_target;
        outlier_prior_prob=outlier_prior_prob,
        max_iters=max_iters,
        max_population=max_population,
        prior_R=cu(rescaled_R_hint), prior_t=cu(rescaled_t_hint), Sigma_inv_R=cu(Sigma_inv_R), Sigma_inv_t=cu(Sigma_inv_t),
        gauss_newton_max_iters = 9
    )
    final_R = opt_R
    final_t = opt_t / scaling + target_mean - opt_R*source_mean

    return final_R, final_t
end

end