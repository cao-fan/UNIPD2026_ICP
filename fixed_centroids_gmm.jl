module FixedCentroidsGMM

export hard_predict_gmm_kernel

using StaticArrays, LinearAlgebra, Random, CUDA

function sample_centroids(point_cloud, k)
    rand_idxs = first(randperm(size(point_cloud,2)),k)
    return rand_idxs
end

function classify_kernel!(mask, data, centroids)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    M = size(centroids,2)
    N = size(data,2)

    indices = index:stride:N
    
    for i in indices
        v = SVector{3}(view(data,:,i))
        min_dist = Inf32
        min_idx = 1
        for j in axes(centroids,2)
            u = SVector{3}(view(centroids,:,j))
            d = v-u
            dist = dot(d,d)
            if dist < min_dist
                min_dist = dist
                min_idx = j
            end
        end
        closest_centroid_idx = min_idx

        mask[i] = closest_centroid_idx
    end
    return nothing
end
function launch_classify_kernel!(mask, data, centroids)
    kernel = @cuda launch=false classify_kernel!(mask, data, centroids)
    config = launch_configuration(kernel.fun)
    threads = min(length(mask), config.threads)
    blocks = cld(length(mask), threads)
    
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks classify_kernel!(mask, data, centroids)
    end
    return nothing
end

function compute_covs_kernel!(covs, counters, centroids, mask, data)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    N = size(data,2)

    indices = index:stride:N
    
    for i in indices
        centroid_idx = mask[i]
        v = SVector{3}(view(data,:,i))
        c = SVector{3}(view(centroids,:,centroid_idx))
        d = v - c
        D = d * transpose(d)

        for cidx in CartesianIndices(D)
            l = cidx[1]
            k = cidx[2]
            CUDA.@atomic covs[l,k,centroid_idx] = covs[l,k,centroid_idx] + D[l,k]
        end
        CUDA.@atomic counters[centroid_idx] = counters[centroid_idx] + 1
    end
    return nothing
end

function launch_compute_covs!(covs, counters, centroids, mask, data)
    kernel = @cuda launch=false compute_covs_kernel!(covs, counters, centroids, mask, data)
    config = launch_configuration(kernel.fun)
    threads = min(size(data,2), config.threads)
    blocks = cld(size(data,2), threads)
    covs .= 0f0
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks compute_covs_kernel!(covs, counters, centroids, mask, data)
    end
    return nothing
end

function clip_cov(cov; min_eigval=1e-8, max_eigval=1.0)
    F = eigen(Symmetric(cov))
    cl = maximum(F.values)
    S = diagm(clamp.(F.values, max(cl/100,min_eigval), min(max_eigval,cl)))
    return Symmetric(F.vectors * S * transpose(F.vectors))
end

function post_process_covs!(covs, counters, clip_cov::F) where F
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    N = size(covs,3)

    indices = index:stride:N
    
    for i in indices
        cov = SMatrix{3,3}(@view(covs[:,:,i]))
        covs[:,:,i] .= cov ./ counters[i]
    end
    return nothing
end
function launch_postprocess_covs!(covs, counters, _clip_cov::F = clip_cov) where F
    kernel = @cuda launch=false post_process_covs!(covs, counters, _clip_cov)
    config = launch_configuration(kernel.fun)
    threads = min(size(covs,3), config.threads)
    blocks = cld(size(covs,3), threads)

    CUDA.@sync begin
        @cuda threads=threads blocks=blocks post_process_covs!(covs, counters, _clip_cov)
    end
    return nothing
end

function compute_P!(P, centroids, data, covs, p_z)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    indices = index:stride:length(P)
    
    CI = CartesianIndices(P)

    for k in indices
        pair = CI[k]
        j = pair[1]
        i = pair[2]

        C = SMatrix{3,3}(@view(covs[:,:,j]))
        μ = SVector{3}(view(centroids,:,j))
        y = SVector{3}(view(data,:,i))
        
        P[j,i] = normal_pdf(y,μ,C) * p_z[j]
    end
    return nothing
end
function launch_compute_P!(P, centroids, data, covs, p_z)
    kernel = @cuda launch=false compute_P!(P, centroids, data, covs, p_z)
    config = launch_configuration(kernel.fun)
    threads = min(length(P), config.threads)
    blocks = cld(length(P), threads)

    CUDA.@sync begin
        @cuda threads=threads blocks=blocks compute_P!(P, centroids, data, covs, p_z)
    end
    row_norm = sum(P;dims=1)
    P .= P ./ row_norm
    return nothing
end

function compute_P_z!(P, p_z)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    indices = index:stride:length(p_z)

    for k in indices
        
        s = 0f0
        c = 0
        for i in axes(P,2)
            s += P[k,i]
            c += 1
        end
        p_z[k] = s / c
    end
    return nothing
end
function launch_compute_P_z!(P, p_z)
    kernel = @cuda launch=false compute_P_z!(P, p_z)
    config = launch_configuration(kernel.fun)
    threads = min(length(p_z), config.threads)
    blocks = cld(length(p_z), threads)

    CUDA.@sync begin
        @cuda threads=threads blocks=blocks compute_P_z!(P, p_z)
    end
    return nothing
end

function compute_soft_covs!(covs, P, centroids, data)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    indices = index:stride:size(centroids,2)

    C =  SMatrix{3,3,Float32}(1,2,3,4,5,6,7,8,9)
    for k in indices
        centroid = SVector{3}(view(centroids,:,k))
        s = 0f0
        C = zero(C)
        for i in axes(P,2)
            y = SVector{3}(view(data,:,i))
            v = y - centroid
            C += P[k,i] * v * transpose(v)
            s += P[k,i]
        end
        covs[:,:,k] .= C ./ s
    end
    return nothing
end
function launch_compute_soft_covs!(covs, P, centroids, data)
    kernel = @cuda launch=false compute_soft_covs!(covs, P, centroids, data)
    config = launch_configuration(kernel.fun)
    threads = min(size(centroids,2), config.threads)
    blocks = cld(size(centroids,2), threads)

    CUDA.@sync begin
        @cuda threads=threads blocks=blocks compute_soft_covs!(covs, P, centroids, data)
    end
    return nothing
end

"""
    hard_predict_gmm_kernel(point_cloud, k = 100)

Given a point cloud, it estimates a GMM distribution with <=k centroids. 
The covariance matrix of a centroid j is computed using the nearest points to the centroid j.
Isolated centroids are filtered out and the covariances are guaranteed to be positive definite(wrapping them in Symmetric is still required).
Returns a tuple (centroids, covariances, mixing_probs):
    *centroids* has dimension (3,N)
    *covariances* has dimension (3,3,N)
    *mixing_probs* has dimension (N,)
"""
function hard_predict_gmm_kernel(point_cloud, k = 100)
    # Take K points at random to be centroids
    centroid_indices = sample_centroids(point_cloud, k)
    training_mask = trues(size(point_cloud,2))
    training_mask[centroid_indices] .= false
    
    centroids_gpu = point_cloud[:,centroid_indices]# |> cu
    training_data_gpu = point_cloud[:,training_mask]# |> cu
    # Classify the training data with the nearest centroid
    classification_mask = CUDA.ones(Int32,size(training_data_gpu,2))
    launch_classify_kernel!(classification_mask, training_data_gpu, centroids_gpu)
    # Compute the covariances
    covs = CUDA.zeros(Float32,3,3,size(centroids_gpu,2))
    counters = CUDA.zeros(Int32,size(centroids_gpu,2))
    launch_compute_covs!(covs, counters, centroids_gpu, classification_mask, training_data_gpu)
    # Normalize covs
    launch_postprocess_covs!(covs,counters)
    # Normalize counters to get the prior probabilities
    counters_sum = sum(counters)
    prior_probabilities = counters ./ counters_sum
    # Filter out the isolated centroids/nonpositive definite covariances
    good_indices = findall(x->!any(isnan.(x)) && ComposedFunction(isposdef,Symmetric)(x), eachslice(covs;dims=3))
    centroids_gpu = centroids_gpu[:,good_indices]
    covs = covs[:,:,good_indices]
    prior_probabilities = prior_probabilities[good_indices]
    return centroids_gpu, covs, prior_probabilities
end

end