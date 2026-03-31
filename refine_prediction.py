import torch
import numpy as np
import math
from scipy import optimize
import pdb

def f(x, a, b, c, d):
    """Function for the optimization solver."""
    return np.sum(a * b * np.exp(-1 * x/c)) - d

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(100-correct_k.mul_(100.0 / batch_size))
        return res

def opt_solver(probs, target_distb, num_iter=10, th=1e-5, num_newton=30):
    """Optimization solver to refine predictions."""
    weights = torch.ones(probs.shape[0])
    N, K = probs.size(0), probs.size(1)

    A, w, lam, nu, r, c = probs.numpy(), weights.numpy(), np.ones(N), np.ones(K), np.ones(N), target_distb.numpy()
    A_e = A / math.e
    X = np.exp(-1 * lam / w)
    Y = np.exp(-1 * nu.reshape(1, -1) / w.reshape(-1, 1))
    prev_Y = np.zeros(K)
    X_t, Y_t = X, Y

    for n in range(num_iter):
        denom = np.sum(A_e * Y_t, 1)
        X_t = r / denom
        Y_t = np.zeros(K)
        for i in range(K):
            Y_t[i] = optimize.newton(f, prev_Y[i], maxiter=num_newton, args=(A_e[:, i], X_t, w, c[i]), tol=th)
        prev_Y = Y_t
        Y_t = np.exp(-1 * Y_t.reshape(1, -1) / w.reshape(-1, 1))

    denom = np.sum(A_e * Y_t, 1)
    X_t = r / denom
    M = torch.Tensor(A_e * X_t.reshape(-1, 1) * Y_t)

    return M

def load_tensors():
    """Loads the tensor data from files."""
    # tensor1 = torch.load('train_cifar10_resnetall_outputs_tensor.pt')
    # tensor2 = torch.load('train_cifar10_resnetall_targets_tensor.pt')
    # tensor3 = torch.load('forget_cifar10_resnetall_outputs_tensor.pt')
    # tensor4 = torch.load('forget_cifar10_resnetall_targets_tensor.pt')
    
    # tensor1 = torch.load('trainall_outputs_tensor.pt')
    # tensor2 = torch.load('trainall_targets_tensor.pt')
    # tensor3 = torch.load('forgetall_outputs_tensor.pt')
    # tensor4 = torch.load('forgetall_targets_tensor.pt')
    
    # tensor1 = torch.load('pt/train_cifar10_selective_allcnnall_outputs_tensor.pt')
    # tensor2 = torch.load('pt/train_cifar10_selective_allcnnall_targets_tensor.pt')
    # tensor3 = torch.load('pt/forget_cifar10_selective_allcnnall_outputs_tensor.pt')
    # tensor4 = torch.load('pt/forget_cifar10_selective_allcnnall_targets_tensor.pt')

    # tensor1 = torch.load('pt/train_lacuna10_allcnnall_outputs_tensor.pt')
    # tensor2 = torch.load('pt/train_lacuna10_allcnnall_targets_tensor.pt')
    # tensor3 = torch.load('pt/forget_lacuna10_allcnnall_outputs_tensor.pt')
    # tensor4 = torch.load('pt/forget_lacuna10_allcnnall_targets_tensor.pt')

    tensor1 = torch.load('train_lacuna10_allcnnall_outputs_tensor.pt')
    tensor2 = torch.load('train_lacuna10_allcnnall_targets_tensor.pt')
    tensor3 = torch.load('forget_lacuna10_allcnnall_outputs_tensor.pt')
    tensor4 = torch.load('forget_lacuna10_allcnnall_targets_tensor.pt')
   
    return tensor1, tensor2, tensor3, tensor4

def refine_predictions(tensor1, tensor2, tensor3, tensor4):
    """Refines predictions using the optimization solver."""
    
    probabilities_tensor1 = torch.softmax(tensor1, dim=1)
    probabilities_tensor3 = torch.softmax(tensor3, dim=1)
    random_values_tensor = torch.rand_like(probabilities_tensor3)
    random_probabilities_tensor = random_values_tensor / random_values_tensor.sum(dim=1, keepdim=True)
    probabilities = torch.cat((probabilities_tensor1, random_probabilities_tensor), dim=0)

    #uniform_probabilities_tensor3 = torch.full_like(probabilities_tensor3, 1 / probabilities_tensor3.size(1)) #may not need tensor3. 
    #probabilities = torch.cat((probabilities_tensor1, uniform_probabilities_tensor3), dim=0)  
    target_tensor = torch.cat((tensor2, tensor4), dim=0)
    target_distb = probabilities.sum(dim=0)
    probabilities = torch.cat((probabilities_tensor1, random_probabilities_tensor), dim=0)
    """target_distb[1:] += target_distb[0] * 0.5 / (len(target_distb) - 1)
    target_distb[0] *= 0.5"""
    refined_prediction = probabilities
    refined_prediction = opt_solver(probabilities.cpu(), target_distb.cpu())
    return refined_prediction

def evaluate_accuracy(tensor1, tensor2, tensor3, tensor4, refined_prediction):
    """Evaluates and prints the accuracy before and after refinement."""
    print(accuracy(tensor1, tensor2))
    print(accuracy(tensor3, tensor4))
    print(accuracy(refined_prediction[:tensor1.shape[0], :].cuda(), tensor2))
    print(accuracy(refined_prediction[tensor1.shape[0]:, :].cuda(), tensor4))
    """pdb.set_trace()"""
    

def main():
    """Main function to run the defined operations."""
    tensor1, tensor2, tensor3, tensor4 = load_tensors()
    refined_prediction = refine_predictions(tensor1, tensor2, tensor3, tensor4)
    evaluate_accuracy(tensor1, tensor2, tensor3, tensor4, refined_prediction)

if __name__ == "__main__":
    main()