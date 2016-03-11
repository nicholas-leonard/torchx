
function torchx.recursiveResizeAs(t1,t2)
   if torch.type(t2) == 'table' then
      t1 = (torch.type(t1) == 'table') and t1 or {t1}
      for key,_ in pairs(t2) do
         t1[key], t2[key] = torchx.recursiveResizeAs(t1[key], t2[key])
      end
   elseif torch.isTensor(t2) then
      t1 = torch.isTensor(t1) and t1 or t2.new()
      t1:resizeAs(t2)
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end

function torchx.recursiveSet(t1,t2)
   if torch.type(t2) == 'table' then
      t1 = (torch.type(t1) == 'table') and t1 or {t1}
      for key,_ in pairs(t2) do
         t1[key], t2[key] = torchx.recursiveSet(t1[key], t2[key])
      end
   elseif torch.isTensor(t2) then
      t1 = torch.isTensor(t1) and t1 or t2.new()
      t1:set(t2)
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end

function torchx.recursiveCopy(t1,t2)
   if torch.type(t2) == 'table' then
      t1 = (torch.type(t1) == 'table') and t1 or {t1}
      for key,_ in pairs(t2) do
         t1[key], t2[key] = torchx.recursiveCopy(t1[key], t2[key])
      end
   elseif torch.isTensor(t2) then
      t1 = torch.isTensor(t1) and t1 or t2.new()
      t1:resizeAs(t2):copy(t2)
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end

function torchx.recursiveAdd(t1, t2)
   if torch.type(t2) == 'table' then
      t1 = (torch.type(t1) == 'table') and t1 or {t1}
      for key,_ in pairs(t2) do
         t1[key], t2[key] = torchx.recursiveAdd(t1[key], t2[key])
      end
   elseif torch.isTensor(t1) and torch.isTensor(t2) then
      t1:add(t2)
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end

function torchx.recursiveTensorEq(t1, t2)
   if torch.type(t2) == 'table' then
      local isEqual = true
      if torch.type(t1) ~= 'table' then
         return false
      end
      for key,_ in pairs(t2) do
          isEqual = isEqual and torchx.recursiveTensorEq(t1[key], t2[key])
      end
      return isEqual
   elseif torch.isTensor(t2) and torch.isTensor(t2) then
      local diff = t1-t2
      local err = diff:abs():max()
      return err < 0.00001
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(t1).." and "..torch.type(t2).." instead")
   end
end

function torchx.recursiveNormal(t2)
   if torch.type(t2) == 'table' then
      for key,_ in pairs(t2) do
         t2[key] = torchx.recursiveNormal(t2[key])
      end
   elseif torch.isTensor(t2) then
      t2:normal()
   else
      error("expecting tensor or table thereof. Got "
           ..torch.type(t2).." instead")
   end
   return t2
end

function torchx.recursiveFill(t2, val)
   if torch.type(t2) == 'table' then
      for key,_ in pairs(t2) do
         t2[key] = torchx.recursiveFill(t2[key], val)
      end
   elseif torch.isTensor(t2) then
      t2:fill(val)
   else
      error("expecting tensor or table thereof. Got "
           ..torch.type(t2).." instead")
   end
   return t2
end

function torchx.recursiveType(param, type_str)
   if torch.type(param) == 'table' then
      for i = 1, #param do
         param[i] = torchx.recursiveType(param[i], type_str)
      end
   else
      if torch.typename(param) and 
        torch.typename(param):find('torch%..+Tensor') then
         param = param:type(type_str)
      end
   end
   return param
end

function torchx.recursiveSum(t2)
   local sum = 0
   if torch.type(t2) == 'table' then
      for key,_ in pairs(t2) do
         sum = sum + torchx.recursiveSum(t2[key], val)
      end
   elseif torch.isTensor(t2) then
      return t2:sum()
   else
      error("expecting tensor or table thereof. Got "
           ..torch.type(t2).." instead")
   end
   return sum
end

function torchx.recursiveNew(t2)
   if torch.type(t2) == 'table' then
      local t1 = {}
      for key,_ in pairs(t2) do
         t1[key] = torchx.recursiveNew(t2[key])
      end
      return t1
   elseif torch.isTensor(t2) then
      return t2.new()
   else
      error("expecting tensor or table thereof. Got "
           ..torch.type(t2).." instead")
   end
end

function torchx.recursiveIndex(res, src, dim, indices)
   if torch.type(src) == 'table' then
      res = (torch.type(res) == 'table') and res or {res}
      for key,_ in pairs(src) do
         res[key] = torchx.recursiveIndex(res[key], src[key], dim, indices)
      end
   elseif torch.isTensor(src) then
      res = torch.isTensor(res) and res or src.new()
      res:index(src, dim, indices)
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(res).." and "..torch.type(src).." instead")
   end
   return res 
end

-- get the batch size (i.e. size of first dim for a nested tensor)
function torchx.recursiveBatchSize(input)
   if torch.type(input) == 'table' then
      return torchx.recursiveBatchSize(input[1])
   else
      assert(torch.isTensor(input))
      return input:size(1)
   end
end

function torchx.recursiveSize(input, excludedim)
   local res
   if torch.type(input) == 'table' then
      res = {}
      for k,v in pairs(input) do
         res[k] = torchx.recursiveSize(v, excludedim)
      end
   else
      assert(torch.isTensor(input))
      res = input:size():totable()
      if excludedim then
         table.remove(res, excludedim)
      end
   end
   return res
end

function torchx.recursiveSub(src, start, stop)
   local res
   if torch.type(src) == 'table' then
      res = {}
      for key,_ in pairs(src) do
         res[key] = torchx.recursiveSub(src[key], start, stop)
      end
   elseif torch.isTensor(src) then
      res = src:sub(start, stop)
   else
      error("expecting nested tensors or tables. Got "..torch.type(src).." instead")
   end
   return res
end
