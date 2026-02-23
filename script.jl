using Rotations

@info "Loading the dependencies..."
include("solve_point_to_plane.jl")
using .SoftICP
include("point_cloud_utils.jl")
using .PointCloudUtils


@info "Loading the data..."
bunny_0 = read_ply("bunny/bun000.ply");
bunny_0_covs = compute_covariances_from_unstructured_point_cloud(bunny_0,k=Val(8));

bunny_1 = read_ply("bunny/bun045.ply");
bunny_1_covs = compute_covariances_from_unstructured_point_cloud(bunny_1,k=Val(8));

source          = bunny_1;#surface_0#bunny_1
target          = bunny_0;#surface_1#bunny_0
source_covs     = bunny_1_covs;#surface_0_covs
target_covs     = bunny_0_covs;#surface_1_covs
THRESHOLD       = 0.1;#10.#0.04


@info "Showing the initial alignment..."
plt = plot_pc(target;markersize=1.5)
plot_pc!(source;markersize=1.5)
display(plt)
println("Input 'continue' to run the registration algorithm")
rd = readline()
if rd == "continue"
    @info "Setting up the solver..."
    final_R,final_t = SoftICP.register(source,target;
        outlier_prior_prob = 0.,
        R_hint = RotXYZ(deg2rad.([0,0,0])...), # True is about -1,34,0.5
        sigmas_R_rad=deg2rad.([0.,0.,0.]),
        t_hint = [0.0,0.000,0.0], # True is about -0.052,-0.0004,-0.011
        sigmas_t = [0.0,0.0,0.0],
        max_iters=100,
        max_population=250
    )
    @info "Showing the alignment results:"
    println("Rotation(Intrinsic Euler XYZ representation):")
    display(rad2deg.(Rotations.params(RotXYZ(RotMatrix3(final_R)))))
    println("Translation:")
    display(final_t)
    @info "Showing final alignment:"
    plt = plot_pc(target;markersize=1.5)
    # plot_pc!(source;markersize=1.5)
    transformed_source = final_R * source .+ final_t
    plot_pc!(transformed_source;markersize=1.5)
    display(plt)

end
