import torch

def fn1():
    a = torch.tensor([-2., 3.], requires_grad=True, device='cuda')
    Q = torch.abs(a)
    external_grad = torch.tensor([1., 1.], device='cuda')
    Q.backward(gradient=external_grad)
    
    # expected [-1, 1] when abs works
    print("abs: ", a.grad)

def fn2():
    a = torch.tensor([-2., 3.], requires_grad=True, device='cuda')
    Q = torch.pow(a, 2)
    external_grad = torch.tensor([1., 1.], device='cuda')
    Q.backward(gradient=external_grad)
    
    # expected [-4, 6] when pow works
    print("pow: ",a.grad)

if __name__ == "__main__":
    torch.cuda.is_available()
    fn1()
    fn2()

