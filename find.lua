

function torch.find(tensor, val, dim)
   local i = 1
   local indice = {}
   if dim then
      assert(tensor:dim() == 2, "torch.find dim arg only supports matrices for now")
      assert(dim == 2, "torch.find only supports dim=2 for now")
            
      local colSize, rowSize = tensor:size(1), tensor:size(2)
      local rowIndice = {}
      tensor:apply(function(x)
            if x == val then
               table.insert(rowIndice, i)
            end
            if i == rowSize then
               i = 1
               table.insert(indice, rowIndice)
               rowIndice = {}
            else
               i = i + 1
            end
         end)
   else
      tensor:apply(function(x)
         if x == val then
            table.insert(indice, i)
         end
         i = i + 1
      end)
   end
   return indice
end

