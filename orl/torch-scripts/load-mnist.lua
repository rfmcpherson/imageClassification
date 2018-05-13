-- Loads and processes MNIST formatted images and labels

local torch = require 'torch'
require 'paths'

-- final dataset 
local mnist = {}
-- number of labels (unique)
local numlabels = 40

-- read images from filename
-- assumes MNIST style
local function readlush(filename)
   local f = torch.DiskFile(filename)
   f:bigEndianEncoding()
   f:binary()

   -- get number of dimensions (images, row, col)
   local ndim = f:readInt() - 0x800
   assert(ndim > 0)

   -- fill dims with num images, row size, col size    
   local dims = torch.LongTensor(ndim)
   for i=1,ndim do
      dims[i] = f:readInt()
      assert(dims[i] > 0)
   end

   -- num elements
   --local nelem = dims:prod(1):squeeze()

   -- read data
   local data = torch.ByteTensor(dims:storage())
   f:readByte(data:storage())
   f:close()
   return data:type('torch.DoubleTensor')
end

-- creates a dataset (obv.)
local function createdataset(dataname, labelname, limit)

   -- load the data and cut off all but the first limit number
  local images = readlush(dataname)
  local label = readlush(labelname)
  images = images[{{1,limit}}]
  label = label[{{1,limit}}]
  assert(images:size(1) == label:size(1))

  local dataset = {images=images, label=label, size=images:size(1)}
  --local labelvector = torch.zeros(10)

  -- normalize the images
  function dataset:normalizeGlobalOld(mean_, std_)
     -- get stats
     local std = std_ or images:std()
     local mean = mean_ or images:mean()
     -- normalize
     images:add(-mean)
     images:mul(1/std)
     return mean, std
  end

  function dataset:normalizeGlobal(mean_, std_)
     if mean_ then
        local negmean = torch.mul(mean_, -1)
        local invstd = torch.pow(std_, -1)

        for i = 1,images:size(1) do
           images[i]:add(negmean)
           images[i]:cmul(invstd)
        end
     else
        -- prep tensors                                     
        local width = images[1]:size(1)
        local height = images[1]:size(2)
        local sum = torch.DoubleTensor(width, height):zero()
        local sum2 = torch.DoubleTensor(width, height):zero()
        local image2

        for i=1, images:size(1) do
           local image = images[i]
           sum:add(image)
           image2 = image:clone()
           image2:cmul(image)
           sum2:add(image2)
        end

        local mean = torch.div(sum, images:size(1))
        
        local mean2 = mean:clone()
        mean2:cmul(mean2)
        local temp = torch.div(sum2, images:size(1))
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

        for i = 1,images:size(1) do
           images[i]:add(negmean)
           images[i]:cmul(invstd)
        end

        return mean, std
     end
   end

  setmetatable(dataset, {__index=function(self, index)
                            assert(index > 0 and index <= self.size)
                            local input = self.images[index]
                            local class = self.label[index]+1
                            local labelv = torch.zeros(numlabels)
                            labelv[class] = 1
                            local example =  {input, labelv}
                            return example
  end})
   
  return dataset
end


-- load training images and labels
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
