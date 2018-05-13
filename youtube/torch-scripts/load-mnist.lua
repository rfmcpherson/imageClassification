-- Loads and processes MNIST formatted images and labels

local torch = require 'torch'
require 'paths'

-- final dataset 
local mnist = {}

-- reads ONE image from filename
-- assumes MNIST style
local function readimage(filename, i)
   local f = torch.DiskFile(filename)
   f:bigEndianEncoding()
   f:binary()

   -- get number of dimensions (images, row, col)
   local ndim = f:readInt() - 0x800
   assert(ndim > 0)

   -- fill dims with num images, row size, col size
   -- normally dims would include the number of images, but not here
   local dims = torch.LongTensor(ndim-1)
 
   images = f:readInt()
   assert(images > 0)

   for j=1,ndim-1 do
      dims[j] = f:readInt()
      assert(dims[j] > 0)
   end
   
   -- jump to desired section of file
   local pos = 4*(ndim+1) -- ndim and dims
   pos = pos + (i-1)*dims[1]*dims[2] + 1 -- size of images
   f:seek(pos)

   -- read data
   local data = torch.ByteTensor(dims:storage())
   f:readByte(data:storage())
   f:close()

   return data:type('torch.DoubleTensor')
end

-- reads ONE label from filename
-- assumes MNIST style
local function readlabel(filename, i)
   local f = torch.DiskFile(filename)
   f:bigEndianEncoding()
   f:binary()

   -- get number of dimensions (images, row, col)
   local ndim = f:readInt() - 0x800
   assert(ndim > 0)

   -- fill dims with num images, row size, col size
   local dims = torch.LongTensor(ndim)

   images = f:readInt()
   assert(images > 0)
   dims[1] = 1

   for i=2,ndim do
      dims[i] = f:readInt()
      assert(dims[i] > 0)
   end
   
   -- jump to desired section of file
   local pos = 4*(1+ndim) -- ndim and dims
   pos = pos + (i-1) + 1-- size of labels
   f:seek(pos)
   
   -- read data
   local data = torch.ByteTensor(dims:storage())
   f:readByte(data:storage())
   f:close()
   return data:type('torch.DoubleTensor')
end

function mnist.loadsingle(set, i, normalize)
   -- load the desired data
   local imagefile
   local labelfile

   if set == "train" then
      imagefile = mnist.trainData.images
      labelfile = mnist.trainData.labels
   elseif set == "valid" then
      imagefile = mnist.validData.images
      labelfile = mnist.validData.labels
   elseif set == "test" then
      imagefile = mnist.testData.images
      labelfile = mnist.testData.labels
   else
      error("Bad input to mnist.loadsingle: " .. set)
   end

   -- load that puppy
   local image = readimage(imagefile, i)
   local label = readlabel(labelfile, i)
 
   -- force one channel
   if FORCE_SINGLE then
      image = image:narrow(1,3,1)
   end
   
   -- normalize
   if normalize then
      image:add(mnist.trainData.negmean)
      image:cmul(mnist.trainData.invstd)
   end

   -- we done
   return {image=image, label=label+1}   
end


function mnist.normalize()
   -- load first image to get dims
   local image = mnist.loadsingle("train", 1, false).image

   local colors
   local width 
   local height

   colors = 1
   width = image:size(1) -- prob width
   height = image:size(2)

   -- prep tensors
   local sum = torch.DoubleTensor(colors, width, height):zero()
   local sum2 = torch.DoubleTensor(colors, width, height):zero()
   local image2

   -- iterate
   for i=1,mnist.trainData.size do
      image = mnist.loadsingle("train", i, false).image
      sum:add(image)
      image2 = image:clone()
      image2:cmul(image)
      sum2:add(image2)

      --[[
      for c=1,colors do
         for w=1,width do
            for h=1,height do
         local val = image[{c,w,h}]
         sum[{c,w,h}] = sum[{c,w,h}] + val
         sum2[{c,w,h}] = sum2[{c,w,h}] + val*val
            end
         end
      end
      ]]--

      if i % 100 == 0 or i == mnist.trainData.size then
         xlua.progress(i,mnist.trainData.size)
      end
   end

   mnist.trainData.mean = torch.div(sum,mnist.trainData.size)

   local mean2 = mnist.trainData.mean:clone()
   mean2:cmul(mean2)
   local temp = torch.div(sum2,mnist.trainData.size)
   temp:csub(mean2)
   mnist.trainData.std = torch.sqrt(temp)

   for i=1,mnist.trainData.std:size(2) do
      for j=1,mnist.trainData.std:size(3) do
         if mnist.trainData.std[1][i][j] == 0 then
            mnist.trainData.std[1][i][j] = 1
         end
      end
   end

   mnist.trainData.negmean = torch.mul(mnist.trainData.mean,-1)
   mnist.trainData.invstd = torch.pow(mnist.trainData.std, -1)

   --print(mnist.trainData.negmean, mnist.trainData.invstd)

   --torch.save("nmean",mnist.trainData.negmean,"ascii")
   --torch.save("istd",mnist.trainData.invstd,"ascii")
   --print("Traing mean", mnist.trainData.mean)
   --print("Training std", mnist.trainData.std)
end


function mnist.setvalues(train, valid, test, trainsize, validsize, testsize)
   -- set the important values

   mnist.trainData = {images=train .. '-images.idx3-ubyte',
                      labels=train .. '-labels.idx1-ubyte',
                      size=trainsize}

   if valid then
      mnist.validData = {images=valid .. '-images.idx3-ubyte',
                         labels=valid .. '-labels.idx1-ubyte',
                         size=validsize}
   end

   mnist.testData = {images=test .. '-images.idx3-ubyte',
                     labels=test .. '-labels.idx1-ubyte',
                     size=testsize}
end

return mnist
