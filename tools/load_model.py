import torch

state = torch.load('./model.pkl', map_location = torch.device('cpu'))['state']

print([energy - state['energies'][0] for energy in state['energies']])
print(state['actions'])