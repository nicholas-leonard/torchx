
function torch.find(tensor, val)
   assert(tensor:dim() == 1)
   local i = 1
   local indice = {}
   tensor:apply(function(x)
         if x == val then
            table.insert(indice, i)
         end
         i = i + 1
      end)
   return torch.LongTensor(indice)
end
