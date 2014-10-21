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

function torchx.test(tests)
   local oldtype = torch.getdefaulttensortype()
   torch.setdefaulttensortype('torch.FloatTensor')
   math.randomseed(os.time())
   mytester = torch.Tester()
   mytester:add(torchxtest)
   mytester:run(tests)
   torch.setdefaulttensortype(oldtype)
end
