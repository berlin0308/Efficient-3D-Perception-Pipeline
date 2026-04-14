import torch
# m = torch.jit.load("pointpillar_traced_compiled.pt")
m = torch.jit.load("pointpillar_traced.pt")
print(type(m))           # torch.jit._trace.TracedModule
print(m.graph)