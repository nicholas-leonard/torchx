
-- recursive map
function torch.remap(t1, t2, f, p1, p2)
   if torch.type(f) ~= 'function' then
      error"Expecting function at argument 3"
   end
   if torch.type(t1) == 'table' then
      t2 = t2 or {}
      for i=1,#t1 do
         t1[i], t2[i] = torch.remap(t1[i], t2[i], f, p1, p2)
      end
   elseif torch.type(t2) == 'table' then
      t1 = t1 or {}
      for i=1,#t2 do
         t1[i], t2[i] = torch.remap(t1[i], t2[i], f, p1, p2)
      end
   elseif torch.isTensor(t1) or torch.isTensor(t2) then
      if not t1 then
         t1 = p1 and p1.new() or t2.new()
      elseif not t2 then
         t2 = (p2 and p2.new()) or (p1 and p1.new()) or t1.new()
      end
      f(t1, t2)
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end

