-- ref.: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
local AM = torch.class("torch.AliasMultinomial") 

function AM:__init(probs)
   self.J, self.q = self:setup(probs)
end

function AM:setup(probs)
   assert(probs:dim() == 1)
   local K = probs:nElement()
   local q = probs.new(K):zero()
   local J = torch.LongTensor(K):zero()

   -- Sort the data into the outcomes with probabilities
   -- that are larger and smaller than 1/K.
   local smaller, larger = {}, {}
   local maxk, maxp = 0, -1
   for kk = 1,K do
      local prob = probs[kk]
      q[kk] = K*prob
      if q[kk] < 1 then
         table.insert(smaller, kk)
      else
         table.insert(larger, kk)
      end
      if maxk > maxp then
         
      end
   end
   
   -- Loop through and create little binary mixtures that
   -- appropriately allocate the larger outcomes over the
   -- overall uniform mixture.
   while #smaller > 0 and #larger > 0 do
      local small = table.remove(smaller)
      local large = table.remove(larger)

      J[small] = large
      q[large] = q[large] - (1.0 - q[small])

      if q[large] < 1.0 then
         table.insert(smaller,large)
      else
         table.insert(larger,large)
      end
   end
   assert(q:min() >= 0)
   if q:max() > 1 then
      q:div(q:max())
   end
   assert(q:max() <= 1)
   if J:min() <= 0 then
      -- sometimes an large index isn't added to J. 
      -- fix it by making the probability 1 so that J isn't indexed.
      local i = 0
      J:apply(function(x)
         i = i + 1
         if x <= 0 then
            q[i] = 1
         end
      end)
   end
   return J, q
end

function AM:draw()
   J = self.J
   q = self.q
   local K  = J:nElement()

   -- Draw from the overall uniform mixture.
   local kk = math.random(1,K)

   -- Draw from the binary mixture, either keeping the
   -- small one, or choosing the associated larger one.
   if math.random() < q[kk] then
      return kk
   else
      return J[kk]
   end
end

function AM:batchdraw(output)
   assert(torch.type(output) == 'torch.LongTensor')
   assert(output:nElement() > 0)
   local J = self.J
   local K  = J:nElement()
   
   self._kk = self._kk or output.new()
   self._kk:resizeAs(output):random(1,K)
   
   self._q = self._q or self.q.new()
   self._q:index(self.q, 1, self._kk:view(-1))
   
   self._mask = self._b or torch.LongTensor()
   self._mask:resize(self._q:size()):bernoulli(self._q)
   
   self.__kk = self.__kk or output.new()
   self.__kk:resize(self._kk:size()):copy(self._kk)
   self.__kk:cmul(self._mask)
   
   -- if mask == 0 then output[i] = J[kk[i]] else output[i] = 0
   
   self._mask:add(-1):mul(-1) -- (1,0) - > (0,1)
   output:view(-1):index(J, 1, self._kk:view(-1))
   output:cmul(self._mask)
   
   -- elseif mask == 1 then output[i] = kk[i]
   
   output:add(self.__kk)
   
   return output
end

