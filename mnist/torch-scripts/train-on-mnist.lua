----------------------------------------------------------------------
-- This script shows how to train different models on the MNIST 
-- dataset, using multiple optimization techniques (SGD, LBFGS)
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

require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
mnist = require './load-mnist'
require 'pl'
require 'paths'

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   -s,--save           (default "logs")      subdirectory to save logs
   -n,--network        (default "")          reload pretrained network
   -m,--model          (default "convnet")   type of model tor train: convnet | mlp | linear
   -f,--full                                 use the full dataset
   -p,--plot                                 plot while training
   -o,--optimization   (default "SGD")       optimization: SGD | LBFGS 
   -r,--learningRate   (default 0.05)        learning rate, for SGD only
   -b,--batchSize      (default 10)          batch size
   -m,--momentum       (default 0)           momentum, for SGD only
   -i,--maxIter        (default 3)           maximum nb of iterations per batch, for LBFGS
   --coefL1            (default 0)           L1 penalty on the weights
   --coefL2            (default 0)           L2 penalty on the weights
   -t,--threads        (default 4)           number of threads
   --train             (default "")          training dataset
   --test              (default "")          test dataset
   --valid             (default "")
   --top               (default 3)           top k to check
   --progress                                print progress bars
   --epochs            (default 200)         how many epochs to run
   --dropout           (default 0)
   --learningRateDecay (default 1e-7)
   --weightDecay       (default 0)
]]

-- Check for training dataset
if opt.train == '' then
   error('--train needs to specify a dataset')
end

-- Check for test dataset
if opt.test == '' then
   error('--test needs to specify a dataset')
end


-- fix seed
torch.manualSeed(1)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- use floats, for SGD
if opt.optimization == 'SGD' then
   torch.setdefaulttensortype('torch.FloatTensor')
end

-- batch size?
if opt.optimization == 'LBFGS' and opt.batchSize < 100 then
   error('LBFGS should not be used with small mini-batches; 1000 is recommended')
end

----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
--
classes = {'1','2','3','4','5','6','7','8','9','10'}

-- geometry: width and height of input images
geometry = {32,32}

if opt.network == '' then
   -- define model to train
   model = nn.Sequential()

   if opt.model == 'convnet' then
      ------------------------------------------------------------
      -- convolutional network 
      ------------------------------------------------------------
      -- mean suppresion -> filter bank -> squashing -> max pooling
      model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
      model:add(nn.LeakyReLU())
      model:add(nn.SpatialMaxPooling(3, 3, 3, 3))

      -- mean suppresion -> filter bank -> squashing -> max pooling
      model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
      model:add(nn.LeakyReLU())
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

      -- standard 2-layer MLP:
      model:add(nn.Reshape(64*2*2))
      model:add(nn.Linear(64*2*2, 200))
      model:add(nn.LeakyReLU())

      if opt.dropout then
         model:add(nn.Dropout(opt.dropout))
      end

      model:add(nn.Linear(200, #classes))
      ------------------------------------------------------------

   elseif opt.model == 'mlp' then
      ------------------------------------------------------------
      -- regular 2-layer MLP
      ------------------------------------------------------------
      model:add(nn.Reshape(1024))
      model:add(nn.Linear(1024, 2048))
      model:add(nn.Tanh())
      model:add(nn.Linear(2048,#classes))
      ------------------------------------------------------------

   elseif opt.model == 'linear' then
      ------------------------------------------------------------
      -- simple linear model: logistic regression
      ------------------------------------------------------------
      model:add(nn.Reshape(1024))
      model:add(nn.Linear(1024,#classes))
      ------------------------------------------------------------

   else
      print('<trainer> unknown model type')
      cmd:text()
      error()
   end
else
   print('<trainer> reloading previously trained network')
   model = torch.load(opt.network)
end

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

-- verbose
print('<mnist> using model:')
print(model)

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion() 
-- REZA: use MSECriterion()
--       Can ignore LogSoftMax (or keep it)
--       Maybe use ReLU as activation function

----------------------------------------------------------------------
-- get/create dataset
--
if opt.full then
   nbTrainingPatches = 60000
   nbTestingPatches = 10000
   nbValidationPatches = 10000
else
   nbTrainingPatches = 2000
   nbValidationPatches = 1000
   nbTestingPatches = 1000
   nbValidationPatches = 1000
   print('<warning> only using 2000 samples to train quickly (use flag -full to use 60000 samples)')
end

-- create training set and normalize
-- trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainData = mnist.traindataset(opt.train .. '-images.idx3-ubyte', 
                               opt.train .. '-labels.idx1-ubyte', 
                               nbTrainingPatches)
mean, std = trainData:normalizeGlobal()

-- create test set and normalize
-- testData = mnist.loadTestSet(nbTestingPatches, geometry)
testData = mnist.testdataset(opt.test .. '-images.idx3-ubyte', 
                             opt.test .. '-labels.idx1-ubyte',
                             nbTestingPatches)
testData:normalizeGlobal(mean, std)

--[[
-- create valid set and normalize
validData =  mnist.testdataset(opt.valid .. '-images.idx3-ubyte', 
                               opt.valid .. '-labels.idx1-ubyte', 
                               nbValidationPatches)
validData:normalizeGlobal(mean, std)
]]--
----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- training function
function train(dataset)
   -- epoch tracker
   epoch = epoch or 1

   -- set model type
   model:training()

   -- local vars
   local time = sys.clock()

   -- set model to training
   model:training()

   -- do one epoch
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,dataset.size,opt.batchSize do
      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset.size) do
         -- load new sample
         local sample = dataset[i]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
         k = k + 1
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- just in case:
         collectgarbage()

         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- evaluate function for complete mini batch
         local outputs = model:forward(inputs)
         local f = criterion:forward(outputs, targets)

         -- estimate df/dW
         local df_do = criterion:backward(outputs, targets)
         model:backward(inputs, df_do)

         -- penalties (L1 and L2):
         if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
            -- locals:
            local norm,sign= torch.norm,torch.sign

            -- Loss:
            f = f + opt.coefL1 * norm(parameters,1)
            f = f + opt.coefL2 * norm(parameters,2)^2/2

            -- Gradients:
            gradParameters:add( sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) )
         end

         -- update confusion
         for i = 1,opt.batchSize do
            confusion:add(outputs[i], targets[i])
         end

         -- return f and df/dX
         return f,gradParameters
      end

      -- optimize on current mini-batch
      if opt.optimization == 'LBFGS' then

         -- Perform LBFGS step:
         lbfgsState = lbfgsState or {
            maxIter = opt.maxIter,
            lineSearch = optim.lswolfe
         }
         optim.lbfgs(feval, parameters, lbfgsState)
       
         -- disp report:
         print('LBFGS step')
         print(' - progress in batch: ' .. t .. '/' .. dataset.size)
         print(' - nb of iterations: ' .. lbfgsState.nIter)
         print(' - nb of function evalutions: ' .. lbfgsState.funcEval)

      elseif opt.optimization == 'SGD' then

         -- Perform SGD step:
         sgdState = sgdState or {
            learningRate = opt.learningRate,
            momentum = opt.momentum,
            learningRateDecay = opt.learningRateDecay, --old:5e-7
            weightDecay = opt.weightDecay
         }
         optim.sgd(feval, parameters, sgdState)
      
         -- disp progress
         if opt.progress then
            xlua.progress(t, dataset.size)
         end

      else
         error('unknown optimization method')
      end
   end

   if opt.progress then
      xlua.progress(dataset.size, dataset.size)
   end

   -- time taken
   time = sys.clock() - time
   time = time / dataset.size
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   --print(confusion)
   confusion:updateValids()
   local totalValid = confusion.totalValid*100
   print("Total Valid: " .. totalValid .. "%")
   trainLogger:add{['% mean class accuracy (train set)'] = totalValid}
   confusion:zero()

   -- save/log current net
   local filename = paths.concat(opt.save, 'mnist.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   --print('<trainer> saving network to '..filename)
   -- torch.save(filename, model)

   -- next epoch
   epoch = epoch + 1
end

-- test function
function test(dataset)
   -- local vars
   local time = sys.clock()
   local correct = 0
   local indices
   local top = torch.Tensor(opt.top):zero()

   -- model type
   model:evaluate()

   -- test over given dataset
   for t = 1,dataset.size,opt.batchSize do

      -- disp progress
      if opt.progress then
         xlua.progress(t, dataset.size)
      end

      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset.size) do
         -- load new sample
         local sample = dataset[i]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
         k = k + 1
      end

      -- test samples
      local preds = model:forward(inputs)

      -- confusion:
      for i = 1,opt.batchSize do
         for j = 1,opt.top do
            NULL, indices = preds[i]:topk(j, true, true)
            if indices[indices:eq(targets[i])]:nDimension() == 1 then
               top[j] = top[j] + 1
            end
         end
         confusion:add(preds[i], targets[i])
      end
   end

   if opt.progress then
      xlua.progress(dataset.size, dataset.size)
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset.size
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   --print(confusion)
   confusion:updateValids()
   local totalValid = confusion.totalValid*100
   print("Total valid: " .. totalValid .. "%")
   for i = 1,opt.top do
      print("Correct in top " .. i .. ": " .. top[i]/dataset.size*100 .. "%") 
      -- print("New Valid: " .. correct/dataset.size*100 .. "%") 
   end
   testLogger:add{['% mean class accuracy (test set)'] = totalValid}
   confusion:zero()
end

----------------------------------------------------------------------
-- and train!
--
while true do
   -- train/test
   print("")
   print('<trainer> on testing set:')
   train(trainData)
   --print('<trainer> on validation set:')
   --test(validData)
   print('<trainer> on training set:')
   test(testData)

   -- plot errors
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      trainLogger:plot()
      testLogger:plot()
   end
   
   -- end
   if epoch > opt.epochs+1 then
      break
   end
end
