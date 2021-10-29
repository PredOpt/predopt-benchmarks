Data generation of shortest path instances as it is done in the article
 Smart “predict, then optimize” (2021). The methodology is as follows:
     
     1. Real model parameters are simulated as a bernoulli(probability = 0.5)
     
     2. Real cost per edge are simulated with the formula 
         c_ij = (1 + 1/math.sqrt(p)*(real+3)**deg )*random.uniform(1-noise, 1+noise)
         
         where p is the number of features of the model, real is the simulated real cost,
         deg controls the misespicification of the linear model by creating a polynomial of 
         higher degree, noise is the half-width perturbation.
   

Function generate_instance receive two parameters:
    K: the amount of instances to generate.
    p: the number of features to generate per instance.
    deg: controls the amount of model misspecification
    noise: random perturbation of the real cost

Function compute_shortest_path(data_file): Solve all the instances in data_file
and store it in a file with a prefix "sol_"

References
Elmachtoub, A. N., & Grigas, P. (2021). Smart “predict, then optimize”. Management Science.