local torchxtest = {}
local precision_forward = 1e-6
local precision_backward = 1e-6
local nloop = 50
local mytester

--e.g. usage: th -ltorchx -e "torchx.test{'treemax','find'}"

function torchxtest.treemax()
	local treeSize = {3,3,2}
	-- 13,14,25                18 = 3x3x2
	--       6,7,12            6 = 3x2
	--           7,5           2 = 2
	local tensor = torch.Tensor{0,0,0,0,0,13, 1,1,1,1,1,10, 0,6, 2,5, 7,5}
	local maxVal, maxIdx = torch.treemax(tensor, treeSize)
	mytester:assert(maxVal == 7, "treemax maxVal 1")
	mytester:assert(maxIdx == 17, "treemax maxIdx 1")
	-- 27,14,25  
	local tensor = torch.Tensor{0,0,0,0,0,27, 1,1,1,1,1,10, 0,6, 2,5, 7,5}
	local maxVal, maxIdx = torch.treemax(tensor, treeSize)
	mytester:assert(maxVal == 27, "treemax maxVal 2")
	mytester:assert(maxIdx == 6, "treemax maxIdx 2")
end

function torchxtest.find()
   local tensor = torch.Tensor{1,2,3,4,5,6,0.6,0,2}
   local indice = torch.LongTensor(torch.find(tensor, 2))
   mytester:assertTensorEq(indice, torch.LongTensor{2,9}, 0.00001, "find (1D) err")
   
   local tensor = torch.Tensor{{1,2,3,4,5},{5,6,0.6,0,2}}
   local indice = torch.find(tensor, 2)
   mytester:assertTableEq(indice, {2,10}, 0.00001, "find (2D) err")
   local indice = torch.find(tensor:t(), 2)
   mytester:assertTableEq(indice, {3,10}, 0.00001, "find (2D transpose) err A")
   local indice = torch.find(tensor:t(), 5)
   mytester:assertTableEq(indice, {2,9}, 0.00001, "find (2D transpose) err B")
   local indice = torch.find(tensor, 2, 2)
   mytester:assertTableEq(indice, {{2},{5}}, 0.00001, "find (2D row-wise) err")
end

function torchxtest.remap()
   local a, b, c, d = torch.randn(3,4), torch.randn(3,4), torch.randn(2,4), torch.randn(1)
   local e, f, g, h = torch.randn(3,4), torch.randn(3,4), torch.randn(2,4), torch.randn(1)
   local t1 = {a:clone(), {b:clone(), c:clone(), {d:clone()}}}
   local t2 = {e:clone(), {f:clone(), g:clone(), {h:clone()}}}
   torch.remap(t1, t2, function(x, y) x:add(y) end)
   mytester:assertTensorEq(a:add(e), t1[1], 0.000001, "error remap a add")
   mytester:assertTensorEq(b:add(f), t1[2][1], 0.000001, "error remap b add")
   mytester:assertTensorEq(c:add(g), t1[2][2], 0.000001, "error remap c add")
   mytester:assertTensorEq(d:add(h), t1[2][3][1], 0.000001, "error remap d add")
   local __, t3 = torch.remap(t2, nil, function(x, y) y:resizeAs(x):copy(x) end)
   mytester:assertTensorEq(e, t3[1], 0.000001, "error remap e copy")
   mytester:assertTensorEq(f, t3[2][1], 0.000001, "error remap f copy")
   mytester:assertTensorEq(g, t3[2][2], 0.000001, "error remap g copy")
   mytester:assertTensorEq(h, t3[2][3][1], 0.000001, "error remap h copy")
   local t4, __ = torch.remap(nil, t2, function(x, y) x:resize(y:size()):copy(y) end, torch.LongTensor())
   mytester:assert(torch.type(t4[1]) == 'torch.LongTensor', "error remap e copy")
   mytester:assert(torch.type(t4[2][1]) == 'torch.LongTensor', "error remap f copy")
   mytester:assert(torch.type(t4[2][2]) == 'torch.LongTensor', "error remap g copy")
   mytester:assert(torch.type(t4[2][3][1]) == 'torch.LongTensor', "error remap h copy")
end

function torchxtest.group()
   local tensor = torch.Tensor{3,4,5,1,2, 5,3,2,1,3, 5,2,3}
   local groups, val, idx = torch.group(tensor)
   mytester:assert(groups[1].idx:size(1) == 2)
   mytester:assert(groups[2].idx:size(1) == 3)
   mytester:assert(groups[3].idx:size(1) == 4)
   mytester:assert(groups[4].idx:size(1) == 1)
   mytester:assert(groups[5].idx:size(1) == 3)
   local tensor2 = tensor:index(1, torch.randperm(tensor:size(1)):long())
   local val2, idx2 = torch.Tensor(), idx:clone()
   local groups = torch.group(val2, idx2, tensor2)
   mytester:assert(groups[1].idx:size(1) == 2)
   mytester:assert(groups[2].idx:size(1) == 3)
   mytester:assert(groups[3].idx:size(1) == 4)
   mytester:assert(groups[4].idx:size(1) == 1)
   mytester:assert(groups[5].idx:size(1) == 3)
   mytester:assertTensorEq(val, val2, 0.00001)
   mytester:assertTensorNe(idx2, idx, 0.00001)
   -- this was failing for me
   local tensor = torch.Tensor{1,2,0}
   local groups, val, idx = torch.group(tensor)
   mytester:assert(groups[1] and groups[2] and groups[0])
end

function torchxtest.concat()
   local tensors = {torch.randn(3,4), torch.randn(3,6), torch.randn(3,8)}
   local res = torch.concat(tensors, 2)
   local res2 = torch.Tensor(3,4+6+8)
   res2:narrow(2,1,4):copy(tensors[1])
   res2:narrow(2,5,6):copy(tensors[2])
   res2:narrow(2,11,8):copy(tensors[3])
   mytester:assertTensorEq(res,res2,0.00001)
   
   local tensors = {torch.randn(2,3,4), torch.randn(2,3,6), torch.randn(2,3,8)}
   local res = torch.concat(tensors, 3)
   local res2 = torch.Tensor(2,3,4+6+8)
   res2:narrow(3,1,4):copy(tensors[1])
   res2:narrow(3,5,6):copy(tensors[2])
   res2:narrow(3,11,8):copy(tensors[3])
   mytester:assertTensorEq(res,res2,0.00001)
   res:zero():concat(tensors, 3)
   mytester:assertTensorEq(res,res2,0.00001)
end

function torchxtest.AliasMultinomial()
   local probs = torch.Tensor(10):uniform(0,1)
   probs:div(probs:sum())

   local a = torch.Timer()
   local am = torch.AliasMultinomial(probs)
   print("setup in "..a:time().real.." seconds")

   a:reset()
   am:draw()
   print("draw in "..a:time().real.." seconds")

   local output = torch.LongTensor(1000, 1000)
   a:reset()
   am:batchdraw(output)
   print("batchdraw in "..a:time().real.." seconds")

   local counts = torch.Tensor(10):zero()
   output:apply(function(x)
      counts[x] = counts[x] + 1
   end)
   
   counts:div(counts:sum())
   
   mytester:assertTensorEq(probs, counts, 0.001)
end

function torchxtest.MultiCudaTensor()
   if not pcall(function() require 'cutorch' end) then
      return
   end
   
   if cutorch.getDeviceCount() < 2 then
      return
   end
   
   local origdevice = cutorch.getDevice()
   
   local inputsize, outputsize = 200, 100
   local weight = torch.CudaTensor(inputsize, outputsize):uniform(0,1)
   local tensors = {
      cutorch.withDevice(1, function() return weight[{{},{1, outputsize/2}}]:clone() end), 
      cutorch.withDevice(2, function() return weight[{{},{(outputsize/2)+1, outputsize}}]:clone() end)
   }
   local mweight = torch.MultiCudaTensor(2, tensors)
   mytester:assert(mweight.catdim == 2)
   
   -- test size
   mytester:assertTableEq(mweight:size():totable(), {inputsize, outputsize}, 0.000001)
   mytester:assert(mweight:size(1) == inputsize)
   mytester:assert(mweight:size(2) == outputsize)
   
   -- test dim
   mytester:assert(mweight:dim() == 2)
   
   -- test transpose
   local mwt = mweight:t()
   mytester:assert(mwt.catdim == 1)
   mytester:assertTableEq(mwt:size():totable(), {outputsize, inputsize}, 0.000001)
   
   -- test index
   local nindex = 3
   local res = torch.CudaTensor()
   local indices = torch.LongTensor(nindex):random(1,inputsize):cuda()
   mweight.index(res, mweight, 1, indices)
   
   local res2 = torch.CudaTensor()
   weight.index(res2, weight, 1, indices)
   
   mytester:assert(res:getDevice() == res2:getDevice())
   mytester:assertTensorEq(res, res2, 0.00001)
   
   mytester:assertTensorEq(weight[{{},{1, outputsize/2}}]:float(), mweight.tensors[1]:float(), 0.00001)
   mytester:assertTensorEq(weight[{{},{(outputsize/2)+1, outputsize}}]:float(), mweight.tensors[2]:float(), 0.00001)
   
   -- test indexAdd
   
   local src = torch.CudaTensor(nindex, outputsize):fill(1)
   
   weight:indexAdd(1, indices, src)
   mweight:indexAdd(1, indices, src)
   
   mytester:assertTensorEq(weight[{{},{1, outputsize/2}}]:float(), mweight.tensors[1]:float(), 0.00001)
   mytester:assertTensorEq(weight[{{},{(outputsize/2)+1, outputsize}}]:float(), mweight.tensors[2]:float(), 0.00001)
   
   -- test add (updateParameters)
   mweight:add(1, mweight)
   weight:add(1, weight)
   
   mytester:assertTensorEq(weight[{{},{1, outputsize/2}}]:float(), mweight.tensors[1]:float(), 0.00001)
   mytester:assertTensorEq(weight[{{},{(outputsize/2)+1, outputsize}}]:float(), mweight.tensors[2]:float(), 0.00001)
   
   -- test mul (updateGradParameters)
   mweight:mul(2)
   weight:mul(2)
   
   mytester:assertTensorEq(weight[{{},{1, outputsize/2}}]:float(), mweight.tensors[1]:float(), 0.00001)
   mytester:assertTensorEq(weight[{{},{(outputsize/2)+1, outputsize}}]:float(), mweight.tensors[2]:float(), 0.00001)
   
   -- test addmm
   local input = torch.CudaTensor(5, outputsize):uniform(0,1)
   local output = torch.CudaTensor(5, inputsize):zero()
   mweight.addmm(output, 0, output, 1, input, mweight:t())
   
   local output2 = output:clone():zero()
   weight.addmm(output2, 0, output2, 1, input, weight:t())
   
   mytester:assertTensorEq(output, output2, 0.0001)
   
   -- test norm 
   local norm = mweight:norm()
   local norm2 = weight:norm()
   mytester:assert(math.abs(norm - norm2) < 0.0001)
   
   -- test zero
   mweight:zero()
   for i=1,2 do
      cutorch.withDevice(i, function() 
         mytester:assert(mweight.tensors[i]:sum() == 0)
      end)
   end
   
   -- test clone
   local mw2 = mweight:clone()
   mytester:assert(mw2.tensors[1]:getDevice() == mweight.tensors[1]:getDevice())
   mytester:assert(mw2.tensors[2]:getDevice() == mweight.tensors[2]:getDevice())
   cutorch.withDevice(1, function() mytester:assertTensorEq(mw2.tensors[1], mweight.tensors[1], 0.000001) end)
   cutorch.withDevice(2, function() mytester:assertTensorEq(mw2.tensors[2], mweight.tensors[2], 0.000001) end)
   
   -- test resizeAs
   mw2.tensors[1]:resize(mw2.tensors[1]:size(1)/2, mw2.tensors[1]:size(2))
   mw2.tensors[2]:resize(mw2.tensors[2]:size(1)/2, mw2.tensors[2]:size(2))
   
   mw2:resizeAs(mweight)
   
   mytester:assertTableEq(mw2.tensors[1]:size():totable(),  mweight.tensors[1]:size():totable(), 0.000001)
   mytester:assertTableEq(mw2.tensors[2]:size():totable(),  mweight.tensors[2]:size():totable(), 0.000001)
   cutorch.withDevice(1, function() mytester:assertTensorEq(mw2.tensors[1], mweight.tensors[1], 0.000001) end)
   cutorch.withDevice(2, function() mytester:assertTensorEq(mw2.tensors[2], mweight.tensors[2], 0.000001) end)
   
   -- test copy
   cutorch.withDevice(1, function() mweight.tensors[1]:uniform(0,1) end)
   cutorch.withDevice(2, function() mweight.tensors[2]:uniform(0,1) end)
   
   mw2:copy(mweight)
   cutorch.withDevice(1, function() mytester:assertTensorEq(mw2.tensors[1], mweight.tensors[1], 0.000001) end)
   cutorch.withDevice(2, function() mytester:assertTensorEq(mw2.tensors[2], mweight.tensors[2], 0.000001) end)
   
   -- test uniform
   mw2:uniform(-2, -1)
   cutorch.withDevice(1, function() mytester:assert(mw2.tensors[1]:min() >= -2 and mw2.tensors[1]:max() <= -1) end)
   cutorch.withDevice(2, function() mytester:assert(mw2.tensors[2]:min() >= -2 and mw2.tensors[2]:max() <= -1) end)
   
   mytester:assert(cutorch.getDevice() == origdevice)
   
end

function torchx.test(tests)
   local oldtype = torch.getdefaulttensortype()
   torch.setdefaulttensortype('torch.FloatTensor')
   math.randomseed(os.time())
   mytester = torch.Tester()
   mytester:add(torchxtest)
   mytester:run(tests)
   torch.setdefaulttensortype(oldtype)
end
