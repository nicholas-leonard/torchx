require 'torch'
require 'paths'
require 'sys'
ffi = require 'ffi'

torchx = {}

torch.include('torchx', 'md5.lua')
torch.include('torchx', 'treemax.lua')
torch.include('torchx', 'find.lua')
torch.include('torchx', 'remap.lua')
torch.include('torchx', 'group.lua')
torch.include('torchx', 'concat.lua')
torch.include('torchx', 'indexdir.lua')

torch.include('torchx', 'test.lua')


