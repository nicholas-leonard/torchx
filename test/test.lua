local torchxtest = {}
local precision_forward = 1e-6
local precision_backward = 1e-6
local nloop = 50
local mytester

--e.g. usage: th -ltorchx -e "torchx.test{'SoftMaxTree','BlockSparse'}"

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
   local indice = torch.find(tensor, 2)
   mytester:assertTensorEq(indice, torch.LongTensor{2,9}, 0.00001, "find err")
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

function torchx.test(tests)
   local oldtype = torch.getdefaulttensortype()
   torch.setdefaulttensortype('torch.FloatTensor')
   math.randomseed(os.time())
   mytester = torch.Tester()
   mytester:add(torchxtest)
   mytester:run(tests)
   torch.setdefaulttensortype(oldtype)
end
