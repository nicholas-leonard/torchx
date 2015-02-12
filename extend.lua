

function torchx.extend(types, funcs, alias)
   for _,type in ipairs(types) do
      local metatable = torch.getmetatable(alias and ('torch.' .. type .. 'Tensor') or type)
      for funcname, func in pairs(funcs) do
         rawset(metatable, funcname, func)
      end
   end
end
