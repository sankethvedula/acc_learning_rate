require "nn"
require "optim"
require "cunn"
require "cutorch"
mnist = require "mnist"
require "MSEUpdatedCriteria"

-- read data here


image_size = 28

cmd = torch.CmdLine()
cmd:text()
cmd:text('MNIST AutoEncoder')
cmd:text()
cmd:text('Options:')
-- General
cmd:option('-lr',1e-4,'learning rate at t=0')
cmd:option('-momentum',1e-4,'weight decay(SGD only)')
cmd:option('-seed',111,'manualSeed set')
cmd:option('-isCuda',true,'if true - use cuda')
cmd:option('-batchSize',20,"batchSize for training")
cmd:option('-nEpochs',5,"Number of epochs to train")
cmd:text()
opt = cmd:parse(arg)


torch.manualSeed(opt.seed)
print({opt})

local function read_data(isTrain)
  if isTrain then
    data = mnist.traindataset().data:float()
    labels = mnist.traindataset().label:float()
    data_1 = torch.Tensor(torch.Tensor(60000,2,28,28))
  else
    data = mnist.testdataset().data:float()
    labels = mnist.testdataset().label:float()
    data_1 = torch.Tensor(torch.Tensor(10000,2,28,28))
  end
  data:add(-127):div(128); -- center data around 0
	data = data:view(data:size(1),1,data:size(2),data:size(3))
  print(data:size())
  data_1[{{},{2},{},{}}] = data
  data_1[{{},{1},{},{}}] = data
  data = data_1
  return data,labels
end

local function create_net()
    local input_size = image_size*image_size
    geometry = {image_size, image_size}

    local fSize = {input_size,400,100,30}
    encoder = nn.Sequential()
    encoder:add(nn.Linear(fSize[1],fSize[2]))
    encoder:add(nn.Tanh())
    encoder:add(nn.Linear(fSize[2],fSize[3]))
    encoder:add(nn.Tanh())
    encoder:add(nn.Linear(fSize[3],fSize[4]))
    encoder:add(nn.Tanh())
    -- decoder
    decoder = nn.Sequential()
    decoder:add(nn.Linear(fSize[4],fSize[3]))
    decoder:add(nn.Tanh())
	  decoder:add(nn.Linear(fSize[3],fSize[2]))
	  decoder:add(nn.Tanh())
    decoder:add(nn.Linear(fSize[2],fSize[1]))

    model = nn.Sequential()
    model:add(nn.View(image_size*image_size))
    model:add(encoder)
    model:add(decoder)
    model:add(nn.View(2,image_size,image_size))

    local criterion = nn.MSEUpdatedCriteria()
    return model, criterion
end


-- Create Models

net, criterion = create_net()

if opt.isCuda then
  net = net:cuda()
  criterion = criterion:cuda()
end

-- Read Data here

trainData, trainLabels = read_data(true)
--print(trainData:size())
--print(trainLabels:size())

testData, testLabels = read_data(false)
--print(testData:size())
--print(testLabels:size())

if opt.isCuda then
  trainData = trainData:cuda()
  trainLabels = trainLabels:cuda()
  testData = testData:cuda()
  testLabels = testLabels:cuda()
end

function train()
  epoch = epoch or 1

  net:training()
  local time = sys.clock()
  total_loss = 0
  num_batches = 0

  for t = 1, trainData:size(1),opt.batchSize do
    num_batches = num_batches + 1
    trainInputs = trainData:narrow(1,t,opt.batchSize)
    trainTargets = trainLabels:narrow(1,t,opt.batchSize)

    params, grads  = net:getParameters()

    if opt.isCuda then
      params = params:cuda()
      grads = grads:cuda()
    end

    local feval = function(x, inputsParam, targetsParam)

      local inputs = inputsParam or trainInputs
      local targets = targetsParam or trainInputs
      grads:zero()
      --print(inputs:size())
      --print(targets:size())
      local outputs = net:forward(inputs)
      --print(outputs:size())
      local f = criterion:forward(outputs, targets)
      local df_do = criterion:backward(outputs, targets)
      net:backward(inputs, df_do)
      --if (t % 5 == 0) then
        return f, grads
      --else
        --return f, grads
      --end
    end

    optimConfig = {
      lr = opt.lr,
      momentum = opt.momentum
    }

    _, errs = optim.sgd(feval, params, optimConfig)
    print(errs[1])
    total_loss = total_loss + errs[1]
  end

  total_loss = total_loss/num_batches
  print("Training error :  ".. total_loss)

end

for epoch = 1, opt.nEpochs do
  print("=== EPOCH "..epoch.."  ===")
  train()
end
