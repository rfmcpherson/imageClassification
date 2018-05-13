local torch = require 'torch'
require 'paths'

local mnist = {}

local function readlush(filename, limit)
   local f = torch.DiskFile(filename)
   f:bigEndianEncoding()
   f:binary()
   local ndim = f:readInt() - 0x800
   assert(ndim > 0)
   local dims = torch.LongTensor(ndim)
   for i=1,ndim do
      dims[i] = f:readInt()
      assert(dims[i] > 0)
   end
   local nelem = dims:prod(1):squeeze()
   local data = torch.ByteTensor(dims:storage())
   f:readByte(data:storage())
   f:close()
   return data:type('torch.DoubleTensor')
   --[{{1,limit}}]
end

local function createdataset(dataname, labelname, limit)
  local data = readlush(dataname, limit)
  data = data[{{1,limit}}]
  local label = readlush(labelname, limit)
  label = label[{{1,limit}}]
  assert(data:size(1) == label:size(1))

  local dataset = {data=data, label=label, size=data:size(1)}
  local labelvector = torch.zeros(10)


  function dataset:normalizeGlobalOld(mean_, std_)
      local std = std_ or data:std()
      local mean = mean_ or data:mean()
      data:add(-mean)
      data:mul(1/std)
      return mean, std
   end

  function dataset:normalizeGlobal(mean_, std_)
     if mean_ then
        local negmean = torch.mul(mean_, -1)
        local invstd = torch.pow(std_, -1)

        for i = 1,data:size(1) do
           data[i]:add(negmean)
           data[i]:cmul(invstd)
        end
     else
        -- prep tensors                                     
        local width = data[1]:size(1)
        local height = data[1]:size(2)
        local sum = torch.DoubleTensor(width, height):zero()
        local sum2 = torch.DoubleTensor(width, height):zero()
        local image2

        for i=1, data:size(1) do
           image = data[i]
           sum:add(image)
           image2 = image:clone()
           image2:cmul(image)
           sum2:add(image2)
        end

        local mean = torch.div(sum, data:size(1))
        
        local mean2 = mean:clone()
        mean2:cmul(mean2)
        local temp = torch.div(sum2, data:size(1))
        temp:csub(mean2)
        local std = torch.sqrt(temp)
                
        -- if no std, don't do anything (done by switching it to 1)
        for i=1,std:size(1) do
           for j=1,std:size(2) do
              if std[i][j] == 0 then
                 std[i][j] = 1
              end
           end
        end

        local negmean = torch.mul(mean, -1)
        local invstd = torch.pow(std, -1)

        for i = 1,data:size(1) do
           data[i]:add(negmean)
           data[i]:cmul(invstd)
        end

        return mean, std
     end
   end


  setmetatable(dataset, {__index=function(self, index)
      assert(index > 0 and index <= self.size)
      local input = self.data[index]
      local class = self.label[index]+1
      local labelv = torch.zeros(10)
      labelv[class] = 1
      local example =  {input, labelv}
      return example
  end})
   
  return dataset
end


function mnist.traindataset(imagefile, labelfile, limit)
   return createdataset(imagefile,
                        labelfile,
                        limit)
end

function mnist.testdataset(imagefile, labelfile, limit)
   return createdataset(imagefile,
                        labelfile,
                        limit)

end

return mnist
