local treeMaxBuffer = {}
function torch.treemax(tensor, treeSize)
   assert(torch.type(treeSize) == 'table')
   assert(tensor:dim() == 1)
   local tmb = treeMaxBuffer[torch.type(tensor)] -- upvalue
   if not tmb then
      tmb = {
         mean = tensor.new(),
         max = tensor.new(),
         idx = torch.LongTensor()
      }
      treeMaxBuffer[torch.type(tensor)] = tmb
   end
   
   local lvl = tensor
   local maxIdx, maxVal = 0, 0
   for i=1,#treeSize do
      lvl = lvl:view(treeSize[i], -1)
      local lvlStride = lvl:size(2)
      if i < #treeSize then
         tmb.mean:mean(lvl, 2)
      end
      tmb.max:max(tmb.idx, tmb.mean:select(2,1), 1)
      local lvlMax, lvlIdx = tmb.max[1], tmb.idx[1]
      lvl = lvl[lvlIdx]
      table.insert(maxPath, lvlIdx)
      maxIdx = maxIdx + (lvlIdx-1)*lvlStride
      maxVal = lvlMax
   end
   return maxVal, maxIdx
end
