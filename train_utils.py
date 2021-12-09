from solver import get_markowitz_constraints_cvx, solve_markowitz_cvx

# c_true are generally the target data
# c_pred are then the model-predicted values
# input and output are expected to be torch tensors
def spo_grad_grb(c_true, c_pred, solver, variables):

    c_spo = (2*c_pred - c_true)
    grad = []
    regret = []
    for i in range(len(c_spo)):    # iterate the batch
        sol_true = solve_markowitz_grb(solver,variables, np.array(c_true[i]))
        sol_spo  = solve_markowitz_grb(solver,variables, np.array(c_spo[i]))
        grad.append(  torch.Tensor(sol_spo - sol_true)  )
        regret.append(  torch.dot(c_true, torch.Tensor(sol_true - sol_pred)  ) )   # this is only for diagnostic / results output

    grad = torch.stack( grad )
    regret = torch.stack( regret )
    return grad, regret

# model is the ML/NN
# optimizer is the torch object
# solver is the CO/LP/QP solver
# variables is the solver's variable handles
def train_fwdbwd_spo_grb(model, optimizer, solver, variables, input_bat, target_bat):

    c_true = target_bat
    c_pred = model(input_bat)
    grad, regret = spo_grad_grb(c_true, c_pred, solver, variables)
    optimizer.zero_grad()
    c_pred.backward(gradient = grad)
    optimizer.step()

    return regret




def spo_grad_cvx(c_true, c_pred, constraints, variables):

    c_spo = (2*c_pred - c_true)
    grad = []
    regret = []
    for i in range(len(c_spo)):    # iterate the batch
        sol_true = solve_markowitz_cvx(constraints,variables, np.array(c_true[i]))
        sol_spo  = solve_markowitz_cvx(constraints,variables, np.array(c_spo[i]))
        grad.append(  torch.Tensor(sol_spo - sol_true)  )
        regret.append(  torch.dot(c_true, torch.Tensor(sol_true - sol_pred)  ) )   # this is only for diagnostic / results output

    grad = torch.stack( grad )
    regret = torch.stack( regret )
    return grad, regret



def train_fwdbwd_spo_cvx(model, optimizer, constraints, variables, input_bat, target_bat):

    c_true = target_bat
    c_pred = model(input_bat)
    grad, regret = spo_grad_cvx(c_true, c_pred, solver, variables)
    optimizer.zero_grad()
    c_pred.backward(gradient = grad)
    optimizer.step()

    return regret



def train_fwdbwd_blackbox_cvx(model, optimizer, blackbox_layer, input_bat, target_bat):

    c_true = target_bat
    c_pred = model(input_bat)
    solver_pred_out = blackbox_layer.apply( c_pred )
    solver_true_out = blackbox_layer.apply( c_true )
    regret = torch.dot( c_true, (solver_true_out - solver_pred_out) )
    optimizer.zero_grad()
    regret.backward()
    optimizer.step()

    return regret


# TODO: the markowitz solving has to be batchified
def BlackboxMarkowitzWrapper(constraints, variables, lambd):

    class BlackboxMarkowitz(torch.autograd.Function):

        @staticmethod
        def forward(ctx, c):
            y = solve_markowitz_cvx(constraints,variables,c)  #make sure this is doing the right thing over the batch
            ctx.save_for_backward( c,y )
            return y

        @staticmethod
        def backward(ctx, grad_output):
            c,y = ctx.saved_tensors
            c_p =  c +  grad_output * lambd
            y_lambd = solve_markowitz_cvx(constraints,variables,c_p)
            # multiply each gradient by the jacobian for the corresponding sample
            # then restack the results to preserve the batch gradients' format
            grad_input = - 1/lambd*(  y - y_lambd  )

            return grad_input

    return BlackboxMarkowitz
