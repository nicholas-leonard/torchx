------------------------------------------------------------------------
--[[ MultiCudaTensor ]]--
-- This experimental tensor is used by the NCEModule in dpnn to 
-- distribute weight/gradWeight over 2 gpus.
-- The MCT only implements the small fraction of use-cases that the
-- NCEModule requires.
------------------------------------------------------------------------
local MCT = torch.class("torch.MultiCudaTensor")
MCT.__noGPU__ = true -- will prevent nn.GPU from switching devices

-- each buffer is indexed by device
local buffers1, buffers2 = {}, {}

function MCT:__init(catdim, tensors)
   self.catdim = catdim or -1
   self.tensors = tensors or {}
end

function MCT:size(dim)
   if not self._size then
      if #self.tensors == 0 then
         self._size = {}
      end
      self._size = self.tensors[1]:size():totable()
      for i=2,#self.tensors do
         self._size[self.catdim] = self._size[self.catdim] + self.tensors[i]:size(self.catdim)
      end
   end
   if dim then
      return self._size[dim]
   end
   return torch.LongStorage(self._size)
end

function MCT:dim()
   return self:size():size()
end

function MCT:zero()
   for i,tensor in ipairs(self.tensors) do
      cutorch.withDevice(tensor:getDevice(), function()
         tensor:zero()
      end)
   end
   return self
end

function MCT:t()
   assert(self:size():size() == 2)
   return self:transpose(1,2)
end

function MCT:transpose(dim1, dim2)
   local dim = self.catdim
   if dim1 == self.catdim then
      dim = dim2
   elseif dim2 == self.catdim then
      dim = dim1
   end
   local tensors = {}
   for i,tensor in ipairs(self.tensors) do
      cutorch.withDevice(tensor:getDevice(), function()
         tensors[i] = tensor:transpose(dim1, dim2)
      end)
   end
   local result = self.new(dim, tensors)
   return result
end

-- self.weight.index(self._weight, self.weight, 1, self.sampleidx:view(-1))
function MCT.index(res, src, dim, indices)
   -- we only support a specific use-case
   assert(torch.type(res) == 'torch.CudaTensor')
   assert(torch.type(src) == 'torch.MultiCudaTensor')
   assert(torch.type(dim) == 'number')
   assert(dim == 1)
   assert(torch.type(indices) == 'torch.CudaTensor' or torch.type(indices) == 'torch.LongTensor')
   assert(indices:dim() == 1)
   assert(src.catdim ~= dim)
   
   local size = src:size()
   size[dim] = indices:size(1)
   res:resize(size)
   
   local start = 1
   for i,srctensor in ipairs(src.tensors) do
      local device = srctensor:getDevice()
      local res_ = res:narrow(src.catdim, start, srctensor:size(src.catdim))
      local res__ = res_
      
      cutorch.withDevice(device, function()
         if device ~= res_:getDevice() then
            buffers2[device] = buffers2[device] or res_.new()
            buffers2[device]:resizeAs(res_):copy(res_)
            res__ = buffers2[device]
         end
         
         if torch.type(indices) == 'torch.CudaTensor' and indices:getDevice() ~= device then
            buffers1[device] = buffers1[device] or indices.new()
            buffers1[device]:resizeAs(indices):copy(indices)
            res__:index(srctensor, dim, buffers1[device])
         else
            res__:index(srctensor, dim, indices)
         end
         
      end)
      
      if device ~= res:getDevice() then
         res_:copy(res__)
      end
      
      start = start + res_:size(src.catdim)
   end
   return res
end

-- self.gradWeight:indexAdd(1, sampleidx, _gradWeight)
function MCT:indexAdd(dim, indices, src)
   assert(torch.type(src) == 'torch.CudaTensor')
   assert(torch.type(dim) == 'number')
   assert(dim == 1)
   assert(self.catdim ~= dim)
   assert(torch.type(indices) == 'torch.CudaTensor' or torch.type(indices) == 'torch.LongTensor')
   
   local start = 1
   for i,tensor in ipairs(self.tensors) do
      local device = tensor:getDevice()
      local src_ = src:narrow(self.catdim, start, tensor:size(self.catdim))
      local src__ = src_
      
      cutorch.withDevice(device, function()
         if device ~= src:getDevice() then
            buffers2[device] = buffers2[device] or src.new()
            buffers2[device]:resizeAs(src_):copy(src_)
            src__ = buffers2[device]
         end
         
         if torch.type(indices) == 'torch.CudaTensor' and indices:getDevice() ~= device then
            buffers1[device] = buffers1[device] or indices.new()
            buffers1[device]:resizeAs(indices):copy(indices)
            tensor:indexAdd(dim, buffers1[device], src__)
         else
            tensor:indexAdd(dim, indices, src__)
         end
      end)
      
      start = start + src_:size(self.catdim)
   end
   
   return self
end

function MCT:add(value, src)
   if not src then
      src = value
      value = 1
   end
   assert(torch.type(src) == 'torch.MultiCudaTensor')
   assert(torch.type(value) == 'number')
   
   for i,srctensor in ipairs(src.tensors) do
      local dstdevice = self.tensors[i]:getDevice()
      local srcdevice = srctensor:getDevice()
      assert(dstdevice == srcdevice)
      cutorch.withDevice(srcdevice, function()
         self.tensors[i]:add(value, srctensor)
      end)
   end
   return self
end

-- momGradParams[i]:mul(momFactor)
function MCT:mul(value)
   for i,tensor in ipairs(self.tensors) do
      cutorch.withDevice(tensor:getDevice(), function() tensor:mul(value) end)
   end
   return self
end

-- self.weight.addmm(self.linout, 0, self.linout, 1, input, self.weight:t())
-- res = (v1 * M) + (v2 * mat1 * mat2)
function MCT.addmm(res, v1, M, v2, mat1, mat2)
   -- we only support a specific use-case
   assert(mat2.catdim == 1)
   assert(torch.type(mat2) == 'torch.MultiCudaTensor')
   assert(torch.type(mat1) == 'torch.CudaTensor')
   assert(torch.type(M) == 'torch.CudaTensor' and torch.pointer(M) == torch.pointer(res))
   assert(torch.type(res) == 'torch.CudaTensor')
   res:mul(v1)
   
   local start = 1
   local lastres = res
   for i,mat2_ in ipairs(mat2.tensors) do
      local mat1_ = mat1:narrow(2, start, mat2_:size(1))
      local device = mat2_:getDevice()
      
      cutorch.withDevice(device, function()
         if device ~= mat1_:getDevice() then
            buffers2[device] = buffers2[device] or mat1_.new()
            buffers2[device]:resizeAs(mat1_):copy(mat1_)
            mat1_ = buffers2[device]
         end
         
         buffers1[device] = buffers1[device] or lastres.new()
         buffers1[device]:resizeAs(res)
         buffers1[device]:mm(mat1_, mat2_)
      end)
      
      local resdevice = res:getDevice()
      if device == resdevice then
         res:add(v2, buffers1[device])
      else
         cutorch.withDevice(resdevice, function()
            buffers1[resdevice] = buffers1[resdevice] or res.new()
            buffers1[resdevice]:resizeAs(res):copy(buffers1[device])
         end)
         res:add(v2, buffers1[resdevice])
      end
      
      start = start + mat2_:size(1)
   end

   assert(start-1 == mat2:size(1))
   return res
end

-- gradParam.new():resizeAs(gradParam):copy(gradParam)
function MCT:resizeAs(src)
   self.catdim = src.catdim
   for i,tensor in ipairs(src.tensors) do
      self.tensors[i] = self.tensors[i] or tensor.new()
      cutorch.withDevice(tensor:getDevice(), function() self.tensors[i]:resizeAs(tensor) end)
   end
   return self
end

function MCT:copy(src)
   for i,tensor in ipairs(src.tensors) do
      self.tensors[i]:copy(tensor)
   end
   return self
end

function MCT:write(file)
   -- Write all values in the object as a table.
   local object = {}
   local tensors = self.tensors
   self.tensors = nil
   for k, v in pairs(self) do
      object[k] = v
   end
   
   file:writeObject(object)
   file:writeObject(#tensors)
   
   for i,tensor in ipairs(tensors) do
      file:writeObject(tensor:getDevice())
      file:writeObject(tensor)
   end
   
   self.tensors = tensors
end

function MCT:read(file)
   local object = file:readObject()
   for k, v in pairs(object) do
      self[k] = v
   end
   
   self.tensors = {}
   
   local N = file:readObject()
   
   for i=1,N do
      local device = file:readObject()
      self.tensors[i] = cutorch.withDevice(device, function() return file:readObject() end)
   end
end

function MCT:clone()
   local f = torch.MemoryFile("rw"):binary()
   f:writeObject(self)
   f:seek(1)
   local clone = f:readObject()
   f:close()
   return clone
end

function MCT:uniform(lower, upper)
   for i,tensor in ipairs(self.tensors) do
      cutorch.withDevice(tensor:getDevice(), function() tensor:uniform(lower, upper) end)
   end
   return self
end

-- math.pow(gradParam:norm(),2)
function MCT:norm(...)
   assert(#{...} == 0)
   local norm = 0
   for i,tensor in ipairs(self.tensors) do
      norm = norm + cutorch.withDevice(tensor:getDevice(), function() return math.pow(tensor:norm(),2) end)
   end
   return math.sqrt(norm)
end

assert(not MCT.storage, "If you ever define storage, you will need to modify Module.sharedClone in dpnn.Module")
