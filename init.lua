require 'torch'

torchx = {}

torch.include('torchx', 'treemax.lua')
torch.include('torchx', 'find.lua')
torch.include('torchx', 'remap.lua')
torch.include('torchx', 'group.lua')
torch.include('torchx', 'concat.lua')

torch.include('torchx', 'test.lua')
