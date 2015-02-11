require 'torch'
require 'paths'
require 'sys'
ffi = require 'ffi'

torchx = {Tensor={}}


torch.include('torchx', 'extend.lua')
torch.include('torchx', 'md5.lua')
torch.include('torchx', 'treemax.lua')
torch.include('torchx', 'find.lua')
torch.include('torchx', 'remap.lua')
torch.include('torchx', 'group.lua')
torch.include('torchx', 'concat.lua')
torch.include('torchx', 'indexdir.lua')

torch.include('torchx', 'test.lua')

local types = {'Byte', 'Char', 'Short', 'Int', 'Long', 'Float', 'Double'}
local Tensor = torchx.Tensor
torchx.Tensor = nil

torchx.extend(types, Tensor, true)

function torchx:cuda()
   torchx:extend({'Cuda'}, Tensor, true)
end

return torchx
