import Random
Random.seed!(42)

using LinearAlgebra, Statistics, Rotations,
      DelimitedFiles, StaticArrays, Tullio,
      Statistics, PlyIO, CUDA

# using Optim, Optimization, OptimizationOptimJL    

using GLMakie#, CairoMakie

include("point_cloud_utils.jl")
include("solve_point_to_plane.jl")
include("fixed_centroids_gmm.jl")



bunny_0 = read_ply("bunny/bun000.ply");
bunny_0_covs = compute_covariances_from_unstructured_point_cloud(bunny_0,k=Val(8));

bunny_1 = read_ply("bunny/bun045.ply");
bunny_1_covs = compute_covariances_from_unstructured_point_cloud(bunny_1,k=Val(8));

source          = bunny_1#surface_0#bunny_1
target          = bunny_0#surface_1#bunny_0
# source_normals  = surface_0_normals
# target_normals  = surface_1_normals
source_covs     = bunny_1_covs#surface_0_covs
target_covs     = bunny_0_covs#surface_1_covs
THRESHOLD       = 0.1#10.#0.04



plt = plot_pc(target;markersize=1.5)
plot_pc!(source;markersize=1.5)
display(plt)
println("Input 'continue' to run the registration algorithm")
readline()
final_R,final_t = register(source,target;
    outlier_prior_prob = 0.,
    R_hint = RotXYZ(deg2rad.([0,0,0])...), # True is about -1,34,0.5
    sigmas_R_rad=deg2rad.([0.,0.,0.]),
    t_hint = [0.0,0.000,0.0], # True is about -0.052,-0.0004,-0.011
    sigmas_t = [0.0,0.0,0.0],
    max_iters=100,
    max_population=250
)
display(rad2deg.(Rotations.params(RotXYZ(RotMatrix3(final_R)))))
display(final_t) 
plt = plot_pc(target;markersize=1.5)
# plot_pc!(source;markersize=1.5)
transformed_source = final_R * source .+ final_t
plot_pc!(transformed_source;markersize=1.5)
display(plt)