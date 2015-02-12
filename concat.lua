
-- torch.concat([res], tensors, [dim])
function torch.concat(result, tensors, dim, index)
   index = index or 1
   if type(result) == 'table' then
      index = dim or 1
      dim = tensors
      tensors = result
      result = tensors[index].new()
   end
   
   assert(type(tensors) == 'table', "expecting table at arg 2")
   
   dim = dim or 1

   local size
   for i,tensor in ipairs(tensors) do
      assert(torch.isTensor(tensor),  "Expecting table of torch.Tensors at arg 2 : "..torch.type(tensor))
      if not size then
         size = tensor:size():totable()
         size[dim] = 0
      end
      for j,v in ipairs(tensor:size():totable()) do
         if j == dim then
            size[j] = (size[j] or 0) + v
         else
            if size[j] and size[j] ~= v then
               error(
                  "Cannot concat dim "..j.." with different sizes: "..
                  (size[j] or 'nil').." ~= "..(v or 'nil')..
                  " for tensor at index "..i, 2
               )
            end
         end
      end
   end
   
   result:resize(unpack(size))
   local start = 1
   for i, tensor in ipairs(tensors) do
      result:narrow(dim, start, tensor:size(dim)):copy(tensor)
      start = start+tensor:size(dim)
   end
   return result
end

torchx.Tensor.concat = torch.concat
