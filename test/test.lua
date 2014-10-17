local torchxtest = {}
local precision_forward = 1e-6
local precision_backward = 1e-6
local nloop = 50
local mytester

--e.g. usage: th -ltorchx -e "torchx.test{'SoftMaxTree','BlockSparse'}"

function torchxtest.treemax()
	
end

function torchx.test(tests)
   local oldtype = torch.getdefaulttensortype()
   torch.setdefaulttensortype('torch.FloatTensor')
   math.randomseed(os.time())
   jac = nn.Jacobian
   mytester = torch.Tester()
   mytester:add(torchxtest)
   mytester:run(tests)
   torch.setdefaulttensortype(oldtype)
end
