module PointCloudUtils

export read_ply, compute_covariances_from_unstructured_point_cloud, plot_pc, plot_pc!, normalize_point_clouds


using StaticArrays,Distributions,NearestNeighbors, Statistics, PlyIO, GLMakie, LinearAlgebra
# using GeometryBasics


"""
    resample_point_cloud(point_cloud::AbstractArray, num_samples::Integer, seed = 123; view=false)
Resamples the point cloud of shape (3,N) if N>num_samples returning a point cloud of shape (3,num_samples) otherwise the point cloud is returned as is.
"""
function resample_point_cloud(point_cloud::AbstractArray, num_samples::Integer; view=false, seed = nothing, return_indices = false)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    source_num_samples = size(point_cloud, 2)
    if source_num_samples > num_samples
        resample_idxs = rand(1:source_num_samples, num_samples)
        if view
            if return_indices
                return @view(point_cloud[:,resample_idxs]), resample_idxs
            else
                return @view point_cloud[:,resample_idxs]
            end
        else
            if return_indices
                return point_cloud[:,resample_idxs], resample_idxs
            else
                return point_cloud[:,resample_idxs]
            end
        end
    else
        if view
            if return_indices
                return point_cloud, collect(axes(point_cloud,2))
            else
                return point_cloud
            end
        else
            if return_indices
                return copy(point_cloud), collect(axes(point_cloud,2))
            else
                return copy(point_cloud)
            end
        end
    end
end


"""
    read_ply(path::AbstractString)
Reads a ply file and returns a (point cloud, normals) tuple having shapes (3,N).
"""
function read_ply(path::AbstractString)
    ply = load_ply(path)
    vertices = transpose([ply["vertex"]["x"];;ply["vertex"]["y"];;ply["vertex"]["z"]])
    pc = Float64.(vertices)
    return pc
end

"""
    compute_covariances_from_unstructured_point_cloud(point_cloud::AbstractArray{T}; k::Val{K} = Val(8)) where {T<:Real,K}

Computes the covariance matrices of a point cloud with the K-nearest neighbors.
"""
function compute_covariances_from_unstructured_point_cloud(point_cloud::AbstractArray{T}; k::Val{K} = Val(8)) where {T<:Real,K}
    if !isa(K, Integer)
        error("K in Val(K) must be an Integer")
    end
    covariances = Array{T,3}(undef,3,3,size(point_cloud,2))
    kdtree = KDTree(point_cloud)
    nn_indices = Vector{Int64}(undef,K+1)
    nn_dists = Vector{T}(undef,K+1)
    centered_nns = Matrix{T}(undef,3,K)
    sortperm_indices = similar(nn_indices)
    # For each point in the point cloud
    for i in axes(point_cloud, 2)
        p = @view point_cloud[:, i]
        # Find the k nearest neighbors
        knn!(nn_indices, nn_dists, kdtree, p, K+1)
        # Sort the nearest neighbors according to their distance, from closest to furthest
        sortperm!(sortperm_indices, nn_dists)
        nn_indices[:] = nn_indices[sortperm_indices]
        indices = @view nn_indices[begin+1:end]
        # Compute the covariance matrix
        centered_nns .= view(point_cloud,:, indices)
        @. centered_nns = centered_nns - p
        covariances[:,:,i] = centered_nns * transpose(centered_nns) / K
    end
    return covariances
end

"""
normalize_point_clouds(source::AbstractArray, target::AbstractArray;dims=2)

Normalizes the point clouds so that they are both contained inside the R^3 unit circle.
Each poing cloud is transformed by X' = γ(X-source_center), Y' = γ(Y-target_center)
where γ is the common scaling coefficient, source_center and target_center are the mean of each point cloud(along the second dimension). 
The point clouds are expected to have shape (3,M) and (3,N) respectively.
Returns
(source_center, target_center, scaling)
"""
function normalize_point_clouds(source::AbstractArray, target::AbstractArray;dims=2)
    source_center = mean(source;dims=2)[:]
    target_center = mean(target;dims=2)[:]

    max_norm_source = mapreduce(x->norm(x-source_center),max,eachslice(source;dims=dims))
    max_norm_target = mapreduce(x->norm(x-target_center),max,eachslice(target;dims=dims))
    scaling = inv(max(max_norm_source,max_norm_target))
    return source_center, target_center, scaling
end


### PLOTTING

"""
Plots the point cloud using Makie
""" 
function plot_pc(pc; markersize=0.75)
    f = Figure(size=(900,900))
    ax = LScene(f[1, 1])
    Makie.scatter!(ax,pc[1,:], pc[2,:], pc[3,:]; markersize = markersize)
    f
end
function plot_pc(pc, normals; arrow_scale = 40, markersize=0.75)
    plot = Makie.scatter(pc[1,:], pc[2,:], pc[3,:]; markersize = markersize)
    Makie.arrows3d!(pc[1,begin:4^4:end],pc[2,begin:4^4:end],pc[3,begin:4^4:end], normals[1,begin:4^4:end],normals[2,begin:4^4:end],normals[3,begin:4^4:end], lengthscale = arrow_scale)
    plot
end
function plot_pc!(pc; markersize=0.75)
    Makie.scatter!(pc[1,:], pc[2,:], pc[3,:]; markersize = markersize)
end

end