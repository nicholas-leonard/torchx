

function torch.find(res, tensor, val, asTable)
   if not torch.isTensor(tensor) then
      asTable = val
      val = tensor
      tensor = res
      if not asTable then
         res = torch.LongTensor()
      end
   end
   assert(tensor:dim() == 1, "torch.find only supports 1D tensors (for now)")
   local i = 1
   local indice = {}
   tensor:apply(function(x)
         if x == val then
            table.insert(indice, i)
         end
         i = i + 1
      end)
   if asTable then
      return indice
   end
   res:resize(#indice)
   i = 0
   res:apply(function()
         i = i + 1
         return indice[i]
      end)
   return res
end

torchx.Tensor.find = torch.find
