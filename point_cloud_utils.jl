using StaticArrays
using GeometryBasics
using Tullio
using Distributions
using NearestNeighbors
using GeometryBasics, Statistics
"""
    read_surface_csv(surface_path::AbstractString)

Reads the CSV file in `surface_path` containing the raw range scans from the profilometer. 

The first row of the file defines the grid spacing along the x and y axis of the profilometer refence frame.
Scan lines are parallel to the y axis, the y axis points downward and the camera points in the direction of the z axis).

From the second row onward each row is a profilometer range scan(x = 0).

Returns a NamedTuple with keys (dx,dy,data)

# Examples
```
dx,dy,data = read_surface_csv(surface_path::AbstractString)
```
"""
function read_surface_csv(surface_path::AbstractString)
    header = readline(surface_path)
    header_items = split(header, ',')
    float_header_items = parse.(Float64,header_items)

    dx, dy = float_header_items
    # Each row represents a laser-range acquisition(x=0 in the profilometer reference frame)
    data = readdlm(surface_path, ',', String, '\n'; header=false, skipstart=1, skipblanks=false)
    # To convert it to xy indexing the matrix is transposed
    data = permutedims(data)
    
    # Empty entries are replaced by NaNs
    nan_parse(s) = isempty(s) ? NaN64 : parse(Float64, s)
    data = nan_parse.(data)
    return (;dx=dx, dy=dy, data=data)
end

"""
    surface_to_points_grid(dx::Real, dy::Real, surface_matrix::AbstractArray)

Converts a 2D surface height matrix into a 3D point grid with x, y coordinates.

Given a surface matrix representing height values, this function creates corresponding
x and y coordinate matrices based on the grid spacing dx and dy, then stacks them
to form a 3D array of 3D points.

Returns a 3D array of shape (3, X_dim, Y_dim) containing the full 3D point grid.
"""
function surface_to_points_grid(dx::Real, dy::Real, surface_matrix::AbstractArray)
    y_num, x_num = size(surface_matrix)
    x_matrix = fill(dx, y_num) * (0:(x_num-1))'
    y_matrix = (0:(y_num-1)) * fill(dy, x_num)'
    return stack([x_matrix,y_matrix,surface_matrix],dims=1) # (3,N,M)
end


"""
    points_grid_to_pc(XYZ::AbstractArray, XYnormals::AbstractArray=nothing)

Converts a 3D point grid to a point cloud by reshaping and filtering out NaN values.

When the normals matrix is provided, both points and normals are reshaped.

Returns either a tuple of (points, normals) or just points, depending on whether normals are provided.
"""
function points_grid_to_pc(XYZ::AbstractArray, XYnormals::AbstractArray)
    points = reshape(XYZ[:,begin+1:end-1,begin+1:end-1], 3,:) # (3,N)
    normals = reshape(XYnormals, 3,:) # (3,N)
    normals_not_nan_mask = BitVector(dropdims(broadcast(~,any(isnan.(normals); dims=1));dims=1))
    not_nan_mask = @. !isnan(points[3,:])
    mask = @. normals_not_nan_mask & not_nan_mask
    return points[:, mask], normals[:, mask]
end

function points_grid_to_pc(XYZ::AbstractArray, ::Nothing)
    points = reshape(XYZ, 3,:) # (3,N)
    not_nan_mask = @. !isnan(points[3,:])
    return points[:, not_nan_mask]
end

points_grid_to_pc(XYZ::AbstractArray) = points_grid_to_pc(XYZ, nothing)

"""
    load_pc(surface_path::AbstractString, return_normals::Bool=true)

Loads the point cloud from the given surface path. NaN values are filtered out. Normals are returned if `return_normals` is true.
The resulting point cloud and normals have each shape (3,N).
"""
function load_pc(surface_path::AbstractString, return_normals::Bool=false)
    result = read_surface_csv(surface_path);
    points_grid = surface_to_points_grid(result...);
    if return_normals == true
        normals_grid = compute_normals_from_grid(points_grid, result.dx, result.dy);
    else
        normals_grid = nothing
    end
    return points_grid_to_pc(points_grid, normals_grid)
end


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
    compute_normals_from_unstructured_point_cloud(point_cloud::AbstractArray; k::Val{K} = Val(8)) where K

Computes the normals of a point cloud with the K-nearest neighbors.
"""
function compute_normals_from_unstructured_point_cloud(point_cloud::AbstractArray; k::Val{K} = Val(8)) where K
    if !isa(K, Integer)
        error("K in Val(K) must be an Integer")
    end
    normals = similar(point_cloud)
    kdtree = KDTree(point_cloud)
    nn_indices = MVector{K+1, Int64}(undef)
    nn_dists = MVector{K+1, eltype(point_cloud)}(undef)
    centered_nns = MMatrix{3,K,eltype(point_cloud)}(undef)
    sortperm_indices = similar(nn_indices)
    # For each point in the point cloud
    for i in axes(point_cloud, 2)
        p = @view point_cloud[:, i]
        knn!(nn_indices, nn_dists, kdtree, p, K+1)
        # Find the k nearest neighbors
        sortperm!(sortperm_indices, nn_dists)
        nn_indices[:] = nn_indices[sortperm_indices]
        nn_dists[:] = nn_dists[sortperm_indices]
        # Compute the covariance matrix and find the principal components
        indices = @view nn_indices[2:K+1]
        centered_nns[:] = view(point_cloud,:, indices)
        centered_nns .= centered_nns .- p
        cov = MMatrix{3,3}(centered_nns * transpose(centered_nns) / K)
        F = eigen(Symmetric(cov))
        V = MMatrix{3,3}(F.vectors)
        if V[3,1] < 0
            V[:,1] = V[:,1]
        end
        normals[:,i] = V[:,1]
    end
    return normals
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


function mesh_cov!(cov_matrix, center = [0,0,0]; z_scaling=3.0)
    F = eigen(Symmetric(cov_matrix))
    R = F.vectors
    # if det(R) < 0 #Flip one axis if determinant is negative
    #     R[:,1] *= -1.0
    # end
    S = diagm(sqrt.(abs.(F.values)) .* z_scaling)

    # T = R * S



    sphere_origin = zeros(Float64,3)
    sphere_radius = one(Float64)
    unit_sphere = GeometryBasics.Sphere{Float64}(sphere_origin, sphere_radius)
    sphere_mesh = GeometryBasics.mesh(unit_sphere)
    vertices = sphere_mesh.position

    transformed_vertices = (Ref(R) .* (Ref(S) .* vertices))
    transformed_vertices = transformed_vertices .+ Ref(center)
    vertices .= transformed_vertices
    ellipsoid_mesh = sphere_mesh
    # Plot ellipsoid
    mesh!(ellipsoid_mesh)
end

function mesh_cov!(dist::MvNormal; z_scaling = 3.0)
    mesh_cov!(dist.Σ, dist.μ;z_scaling = z_scaling)
end
