local Queue = torch.class("torchx.Queue")

function Queue:__init()
   self.first = 0
   self.last = -1
   self.list = {}
end

function Queue:put(value)
   local first = self.first - 1
   self.first = first
   self.list[first] = value
end

function Queue:empty()
   return self.first > self.last
end
 
function Queue:get()
   local last = self.last
   if self:empty() then 
      error("Queue is empty")
   end
   local value = self.list[last]
   self.list[last] = nil  
   self.last = last - 1
   return value
end
