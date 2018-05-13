----------------------------------------------------------------------
-- This script shows how to train different models on the CIFAR
-- dataset, using multiple optimization techniques (SGD, ASGD, CG)
--
-- This script demonstrates a classical example of training
-- well-known models (convnet, MLP, logistic regression)
-- on a 10-class classification problem.
--
-- It illustrates several points:
-- 1/ description of the model
-- 2/ choice of a loss function (criterion) to minimize
-- 3/ creation of a dataset as a simple Lua table
-- 4/ description of training and test procedures
--
-- Clement Farabet
----------------------------------------------------------------------

require 'nn'
require 'optim'
require 'image'

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   --save                  (default "logs")        subdirectory to save logs
   --network               (default "")            reload pretrained network
   --model                 (default "convnet")     model to use
   --full                                          full model
   --seed                  (default 1)             
   --optimization          (default "SGD")        
   --learningRate          (default 1e-3)
   -b,--batchSize          (default 10)
   --weightDecay           (default 0)
   --momentum              (default 0)
   --t0                    (default 1)            start averaging at t0 (ASGD only), in nb of epochs
   --maxIter               (default 5)            maximum nb of iterations for CG and LBFGS
   --threads               (default 4)
   --progress                                     progress
   --infolder              (default "")
   --top                   (default 3)
   --epochs                (default 100)
   --dropout               (default 0)
   --learningRateDecay     (default 1e-7)
]]

if opt.infolder == '' then
   error('Need to specify some input information')
end

-- fix seed
torch.manualSeed(opt.seed)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. opt.threads)

----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
--
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

if opt.network == '' then
   -- define model to train
   model = nn.Sequential()

   if opt.model == 'convnet' then
      ------------------------------------------------------------
      -- convolutional network
      ------------------------------------------------------------
      -- stage 1 : mean+std normalization -> filter bank -> squashing -> max pooling
      model:add(nn.SpatialConvolutionMap(nn.tables.random(3,16,1), 5, 5))
      model:add(nn.LeakyReLU())
      --model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

      -- stage 2 : filter bank -> squashing -> max pooling
      model:add(nn.SpatialConvolutionMap(nn.tables.random(16, 256, 4), 5, 5))
      model:add(nn.LeakyReLU())
      --model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

      -- stage 3 : standard 2-layer neural network
      model:add(nn.Reshape(256*5*5))
      model:add(nn.Linear(256*5*5, 128))
      model:add(nn.LeakyReLU())
      --model:add(nn.Tanh())

      if opt.dropout then
         model:add(nn.Dropout(opt.dropout))
      end

      model:add(nn.Linear(128,#classes))
      ------------------------------------------------------------

   elseif opt.model == 'mlp' then
      ------------------------------------------------------------
      -- regular 2-layer MLP
      ------------------------------------------------------------
      model:add(nn.Reshape(3*32*32))
      model:add(nn.Linear(3*32*32, 1*32*32))
      model:add(nn.Tanh())
      model:add(nn.Linear(1*32*32, #classes))
      ------------------------------------------------------------

   elseif opt.model == 'linear' then
      ------------------------------------------------------------
      -- simple linear model: logistic regression
      ------------------------------------------------------------
      model:add(nn.Reshape(3*32*32))
      model:add(nn.Linear(3*32*32,#classes))
      ------------------------------------------------------------

   else
      print('Unknown model type')
      cmd:text()
      error()
   end
else
   print('<trainer> reloading previously trained network')
   model = nn.Sequential()
   model:read(torch.DiskFile(opt.network))
end

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

-- verbose
print('<cifar> using model:')
print(model)

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

----------------------------------------------------------------------
-- Loading data
--
local function readdata(filename, limit)
   local f = torch.DiskFile(filename)
   f:bigEndianEncoding()
   f:binary()
   local data = torch.ByteTensor(30720000)
   f:readByte(data:storage())
   f:close()
   data = data:reshape(10000,3,32,32)
   return data--:type('torch.DoubleTensor')
end

local function readlabels(filename, limit)
   local f = torch.DiskFile(filename)
   f:bigEndianEncoding()
   f:binary()
   local data = torch.ByteTensor(10000)
   f:readByte(data:storage())
   f:close()
   return data--:type('torch.DoubleTensor')
end

----------------------------------------------------------------------
-- get/create dataset
--
if opt.full then
   trsize = 50000
   tesize = 10000
   vasize = 10000
else
   trsize = 2000
   tesize = 1000
   vasize = 1000
end

-- download dataset
--[[
if not paths.dirp('cifar-10-batches-t7') then
   local www = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'
   local tar = paths.basename(www)
   os.execute('wget ' .. www .. '; '.. 'tar xvf ' .. tar)
end
--]]

-- load dataset
trainData = {
   data = torch.Tensor(50000, 3072),
   labels = torch.Tensor(50000),
   size = function() return trsize end
}
for i = 0,4 do
   --subdata = readdata(opt.infolder .. 'train_data' .. '_' .. (i+1) .. '.raw')
   --sublabels = readlabels(opt.infolder .. 'train_labels'  .. '_' .. (i+1) .. '.raw')
   subdata = readdata(opt.infolder .. 'train_data0' .. '_' .. (i+1) .. '.raw')
   sublabels = readlabels(opt.infolder .. 'train_labels0'  .. '_' .. (i+1) .. '.raw')
   trainData.data[{ {i*10000+1, (i+1)*10000} }] = subdata--:t()
   trainData.labels[{ {i*10000+1, (i+1)*10000} }] = sublabels
end
trainData.labels = trainData.labels + 1

--subdata = readdata(opt.infolder .. 'test_data' .. '.raw')
--sublabels = readlabels(opt.infolder .. 'test_labels' .. '.raw')
subdata = readdata(opt.infolder .. 'test_data0' .. '.raw')
sublabels = readlabels(opt.infolder .. 'test_labels0' .. '.raw')
testData = {
   data = subdata:double(),
   labels = sublabels:double(),
   size = function() return tesize end
}
testData.labels = testData.labels + 1

--[[
subdata = readdata(opt.infolder .. 'valid_data' .. '.raw')
sublabels = readlabels(opt.infolder .. 'valid_labels' .. '.raw')
validData = {
   data = subdata:double(),
   labels = sublabels:double(),
   size = function() return vasize end
}
validData.labels = validData.labels + 1
]]--

-- resize dataset (if using small version)
trainData.data = trainData.data[{ {1,trsize} }]
trainData.labels = trainData.labels[{ {1,trsize} }]

testData.data = testData.data[{ {1,tesize} }]
testData.labels = testData.labels[{ {1,tesize} }]

--[[
validData.data = validData.data[{ {1,vasize} }]
validData.labels = validData.labels[{ {1,vasize} }]
]]--

-- reshape data
trainData.data = trainData.data:reshape(trsize,3,32,32)
testData.data = testData.data:reshape(tesize,3,32,32)
--validData.data = validData.data:reshape(vasize,3,32,32)

----------------------------------------------------------------------
-- preprocess/normalize train/test sets
--

print '<trainer> preprocessing data (color space + normalization)'
collectgarbage()

-- preprocess trainSet
normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
for i = 1,trainData:size() do
   -- rgb -> yuv
   local rgb = trainData.data[i]
   local yuv = image.rgb2yuv(rgb)
   -- normalize y locally:
   yuv[1] = normalization(yuv[{{1}}])
   trainData.data[i] = yuv
end

-- normalize u globally:
mean_u = trainData.data[{ {},2,{},{} }]:mean()
std_u = trainData.data[{ {},2,{},{} }]:std()
trainData.data[{ {},2,{},{} }]:add(-mean_u)
trainData.data[{ {},2,{},{} }]:div(-std_u)

-- normalize v globally:
mean_v = trainData.data[{ {},3,{},{} }]:mean()
std_v = trainData.data[{ {},3,{},{} }]:std()
trainData.data[{ {},3,{},{} }]:add(-mean_v)
trainData.data[{ {},3,{},{} }]:div(-std_v)

-- preprocess testSet
for i = 1,testData:size() do
   -- rgb -> yuv
   local rgb = testData.data[i]
   local yuv = image.rgb2yuv(rgb)
   -- normalize y locally:
   yuv[{1}] = normalization(yuv[{{1}}])
   testData.data[i] = yuv
end
-- normalize u globally:
testData.data[{ {},2,{},{} }]:add(-mean_u)
testData.data[{ {},2,{},{} }]:div(-std_u)
-- normalize v globally:
testData.data[{ {},3,{},{} }]:add(-mean_v)
testData.data[{ {},3,{},{} }]:div(-std_v)

--[[
-- preprocess validSet
for i = 1,validData:size() do
   -- rgb -> yuv
   local rgb = validData.data[i]
   local yuv = image.rgb2yuv(rgb)
   -- normalize y locally:
   yuv[{1}] = normalization(yuv[{{1}}])
   validData.data[i] = yuv
end
-- normalize u globally:
validData.data[{ {},2,{},{} }]:add(-mean_u)
validData.data[{ {},2,{},{} }]:div(-std_u)
-- normalize v globally:
validData.data[{ {},3,{},{} }]:add(-mean_v)
validData.data[{ {},3,{},{} }]:div(-std_v)
]]--
----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- training function
function train(dataset)
   -- epoch tracker
   epoch = epoch or 1

   -- model type
   model:training()

   -- local vars
   local time = sys.clock()
   local trainError = 0

   -- do one epoch
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,dataset:size(),opt.batchSize do

      -- disp progress
      if opt.progress then
         xlua.progress(t, dataset:size())
      end

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local input = dataset.data[i]
         local target = dataset.labels[i]
         table.insert(inputs, input)
         table.insert(targets, target)
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- f is the average of all criterions
         local f = 0

         -- evaluate function for complete mini batch
         for i = 1,#inputs do
            -- estimate f
            local output = model:forward(inputs[i])
            local err = criterion:forward(output, targets[i])
            f = f + err

            -- estimate df/dW
            local df_do = criterion:backward(output, targets[i])
            model:backward(inputs[i], df_do)

            -- update confusion
            confusion:add(output, targets[i])
         end

         -- normalize gradients and f(X)
         gradParameters:div(#inputs)
         f = f/#inputs
         trainError = trainError + f

         -- return f and df/dX
         return f,gradParameters
      end

      -- optimize on current mini-batch
      if opt.optimization == 'CG' then
         config = config or {maxIter = opt.maxIter}
         optim.cg(feval, parameters, config)

      elseif opt.optimization == 'LBFGS' then
         config = config or {learningRate = opt.learningRate,
                             maxIter = opt.maxIter,
                             nCorrection = 10}
         optim.lbfgs(feval, parameters, config)

      elseif opt.optimization == 'SGD' then
         config = config or {learningRate = opt.learningRate,
                             weightDecay = opt.weightDecay,
                             momentum = opt.momentum,
                             learningRateDecay = opt.learningRateDecay --5e-7
                            }
         optim.sgd(feval, parameters, config)

      elseif opt.optimization == 'ASGD' then
         config = config or {eta0 = opt.learningRate,
                             t0 = nbTrainingPatches * opt.t0}
         _,_,average = optim.asgd(feval, parameters, config)

      else
         error('unknown optimization method')
      end
   end

   -- disp progress
   if opt.progress then
      xlua.progress(dataset:size(), dataset:size())
   end

   -- train error
   trainError = trainError / math.floor(dataset:size()/opt.batchSize)

   -- time taken
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   --print(confusion)
   confusion:updateValids()
   local totalValid = confusion.totalValid * 100
   print("Total Valid: " .. totalValid .. "%")
   confusion:zero()

   -- next epoch
   epoch = epoch + 1

   return --trainAccuracy, trainError
end

-- test function
function test(dataset)
   -- local vars
   local testError = 0
   local time = sys.clock()
   local top = torch.Tensor(opt.top):zero()

   -- model type
   model:evaluate()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- test over given dataset
   for t = 1,dataset:size() do

      -- disp progress
      if opt.progress then
         xlua.progress(t, dataset:size())
      end

      -- get new sample
      local input = dataset.data[t]
      local target = dataset.labels[t]

      -- test sample
      local pred = model:forward(input)

      -- confusion:
      for j = 1,opt.top do
         NULL, indices = pred:topk(j, true, true)
         if indices[indices:eq(target)]:nDimension() == 1 then
            top[j] = top[j] + 1
         end
      end
      confusion:add(pred, target)


      -- compute error
      err = criterion:forward(pred, target)
      testError = testError + err
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- testing error estimation
   testError = testError / dataset:size()

   -- print confusion matrix
   --print(confusion)
   confusion:updateValids()
   local totalValid = confusion.totalValid*100
   print("Total valid: " .. totalValid .. "%")
   for i = 1,opt.top do
      print("Correct in top " .. i .. ": " .. top[i]/dataset:size()*100 .. "%") 
      -- print("New Valid: " .. correct/dataset.size*100 .. "%") 
   end
   confusion:zero()

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end

   return --testAccuracy, testError
end

----------------------------------------------------------------------
-- and train!
--
while true do
   -- train/test
   print("")
   print('<trainer> on training set:')
   trainAcc, trainErr = train(trainData)
   print('<trainer> on validation set:')
   --validAcc, validErr = test (validData)
   print('<trainer> on testing set:')
   testAcc,  testErr  = test (testData)

   if epoch >= opt.epochs then
      break
   end
end
