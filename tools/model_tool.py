import torch

state = torch.load('./model.pkl', map_location = torch.device('cpu'))['state']

print([energy - state['energies'][0] for energy in state['energies']])
print(state['actions'])

[0, 2, 3, 0, 0, 0, 0, 0, 3, 0, 2, 0, 3, 0, 2, 0, 2, 0, 3, 0, 2, 3, 0, 2, 0, 0, 0, 3, 3, 3, \
6, 3, 3, 3, 3, 3, 0, 0, 2, 0, 0, 2, 0, 3, 2, 3, 0, 2, 0, 3, 2, 3, 0, 0, 3, 2, 3, 2, 2, 0, 3, \
0, 3, 3, 0, 0, 2, 3, 3, 0, 0, 2, 0, 3, 2, 3, 3, 3, 2, 3, 2, 8, 2, 3, 2, 2, 0, 6, 3, 0, 0, 0, \
2, 3, 3, 8, 1, 4, 2, 2, 3]