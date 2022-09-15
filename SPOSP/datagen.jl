include("./SmartPredictThenOptimize/solver/util.jl")
include("./SmartPredictThenOptimize/oracles/shortest_path_oracle.jl")
using Random, CSV, Tables

n_train_vec = [100; 1000; 5000]
polykernel_degree_vec = [1; 2; 4; 6; 8]
polykernel_noise_half_width_vec = [0; 0.5]
holdout_percent = 0.25
for n_train in n_train_vec
    for polykernel_degree in polykernel_degree_vec
        for polykernel_noise_half_width in polykernel_noise_half_width_vec
            n_test = 10000
            
            n_holdout = round(Int, holdout_percent*n_train)
            num_lambda = 1
            lambda_max = 0
            lambda_min_ratio = 1
            regularization = :lasso
            different_validation_losses = false
            include_rf = true

            p_features = 5
            grid_dim = 5
            num_trials = 50
            rng_seed =  64234
            Random.seed!(rng_seed)

            sources, destinations = convert_grid_to_list(grid_dim, grid_dim)
            d_feasibleregion = length(sources)
            B_true = rand(Bernoulli(0.5), d_feasibleregion, p_features)
            (X_train, c_train) = generate_poly_kernel_data_simple(B_true, n_train, polykernel_degree, polykernel_noise_half_width)
            (X_validation, c_validation) = generate_poly_kernel_data_simple(B_true, n_holdout, polykernel_degree, polykernel_noise_half_width)
            (X_test, c_test) = generate_poly_kernel_data_simple(B_true, n_test, polykernel_degree, polykernel_noise_half_width)

            # Add intercept in the first row of X
            # X_train = vcat(ones(1,n_train), X_train)
            # X_validation = vcat(ones(1,n_holdout), X_validation)
            # X_test = vcat(ones(1,n_test), X_test)

            # # Get Hamming labels
            # (z_train, w_train) = oracle_dataset(c_train, sp_oracle)
            # (z_validation, w_validation) = oracle_dataset(c_validation, sp_oracle)
            # (z_test, w_test) = oracle_dataset(c_test, sp_oracle)

            CSV.write("SyntheticData/TraindataX_N_$(n_train)_noise_$(polykernel_noise_half_width)_deg_$(polykernel_degree).csv", 
            Tables.table(X_train), writeheader=false)
            CSV.write("SyntheticData/TestdataX_N_$(n_train)_noise_$(polykernel_noise_half_width)_deg_$(polykernel_degree).csv", 
            Tables.table(X_test), writeheader=false)
            CSV.write("SyntheticData/ValidationdataX_N_$(n_train)_noise_$(polykernel_noise_half_width)_deg_$(polykernel_degree).csv", 
            Tables.table(X_validation), writeheader=false)
            CSV.write("SyntheticData/Traindatay_N_$(n_train)_noise_$(polykernel_noise_half_width)_deg_$(polykernel_degree).csv", 
            Tables.table(c_train), writeheader=false)
            CSV.write("SyntheticData/Testdatay_N_$(n_train)_noise_$(polykernel_noise_half_width)_deg_$(polykernel_degree).csv", 
            Tables.table(c_test), writeheader=false)
            CSV.write("SyntheticData/Validationdatay_N_$(n_train)_noise_$(polykernel_noise_half_width)_deg_$(polykernel_degree).csv", 
            Tables.table(c_validation), writeheader=false)
        end
    end
end