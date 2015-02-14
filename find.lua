

function torch.find(tensor, val)
   if not torch.isTensor(tensor) then
      asTable = val
      val = tensor
      tensor = res
      if not asTable then
         res = torch.LongTensor()
      end
   end
   local i = 1
   local indice = {}
   tensor:apply(function(x)
         if x == val then
            table.insert(indice, i)
         end
         i = i + 1
      end)
   return indice
end

