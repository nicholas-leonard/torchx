
function torch.group(sorted, index, tensor, samegrp, desc)
   if not torch.isTensor(tensor) then
      desc = tensor
      samegrp = index
      tensor = sorted
      index = nil
      sorted = nil
   end
   assert(torch.isTensor(tensor), 'expecting torch.Tensor for arg 3')
   
   local sorted = sorted or tensor.new()
   assert(torch.type(tensor) == torch.type(sorted), 'expecting torch.Tensor for arg 1 as same type as arg 2')
   
   samegrp = samegrp or function(start_val, val)
      return start_val == val
   end
   assert(torch.type(samegrp) == 'function', 'expecting function for arg 4')
   
   index = index or torch.LongTensor()
   assert(torch.type(index) == 'torch.LongTensor', 'expecting torch.LongTensor for arg 2')
   
   if desc == nil then
      desc = false
   end
   sorted:sort(index, tensor, desc)
   
   local start_idx, start_val = 1, sorted[1]
   local idx = 1
   local groups = {}
   sorted:apply(function(val)
      if not samegrp(start_val, val) then         
         groups[start_val] = {
            idx=index:narrow(1, start_idx, idx-start_idx), 
            val=sorted:narrow(1, start_idx, idx-start_idx)
         }
         start_val = val
         start_idx = idx
      end
      
      idx = idx + 1
      
      if idx-1 == sorted:size(1) then
         groups[start_val] = {
            idx=index:narrow(1, start_idx, idx-start_idx), 
            val=sorted:narrow(1, start_idx, idx-start_idx)
         }
      end
      
   end)

   return groups, sorted, index
end
