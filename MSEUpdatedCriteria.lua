local MSEUpdatedCriteria, parent = torch.class('nn.MSEUpdatedCriteria','nn.Criterion')

function MSEUpdatedCriteria:__init(sizeAverage)
  parent.__init(self)
  if sizeAverage ~= nil then
    self.sizeAverage = sizeAverage
  else
    self.sizeAverage = true
  end
end

function MSEUpdatedCriteria:updateOutput(input, target)
  self.output_tensor = self.output_tensor or input.new(1)
  input.THNN.MSECriterion_updateOutput(
    input:cdata(),
    target:cdata(),
    self.output_tensor:cdata(),
    self.sizeAverage
  )

  gradient_y = nn.Sequential()
  gradient_y:add(nn.SpatialConvolution(1,1,3,3,1,1,1,1))
  weights = gradient_y:get(1).weight
  weights = torch.Tensor{{1,1,1},{0,0,0},{-1,-1,-1}}

  gradient_y:cuda()

  gradient_x = nn.Sequential()
  gradient_x:add(nn.SpatialConvolution(1,1,3,3,1,1,1,1))
  weights = gradient_x:get(1).weight
  weights = torch.Tensor{{1,0,-1},{1,0,-1},{1,0,-1}}

  gradient_x:cuda()

  input1_x = gradient_x:forward(input[{{},{1},{},{}}])
  input2_x = gradient_x:forward(input[{{},{2},{},{}}])

  input1_y = gradient_y:forward(input[{{},{1},{},{}}])
  input2_y = gradient_y:forward(input[{{},{2},{},{}}])

  num_elements = input1_x:size()[1]*input1_x:size()[3]*input1_x:size()[4]*2

  regularize_x = torch.dot(input1_x,input2_x)
  regularize_x = regularize_x/num_elements
  regularize_y = torch.dot(input1_y,input2_y)
  regularize_y = regularize_y/num_elements

  regularize_y = torch.abs(regularize_y)
  regularize_x = torch.abs(regularize_x)

  norm1_x = torch.norm(input1_x)
  norm2_x = torch.norm(input2_x)

  norm1_y = torch.norm(input1_y)
  norm2_y = torch.norm(input2_y)

  normed_regularization_x = regularize_x
  normed_regularization_y = regularize_y

  self.output = self.output_tensor[1] - normed_regularization_y - normed_regularization_x
  return self.output
end

function MSEUpdatedCriteria:updateGradInput(input, target)
  input.THNN.MSECriterion_updateGradInput(
    input:cdata(),
    target:cdata(),
    self.gradInput:cdata(),
    self.sizeAverage
  )

  gradient_x = nn.Sequential()
  gradient_x:add(nn.SpatialConvolution(1,1,3,3,1,1,1,1))
  weights = gradient_x:get(1).weight
  weights = torch.Tensor{{1,0,-1},{1,0,-1},{1,0,-1}}

  gradient_x:cuda()

  input1_x = gradient_x:forward(input[{{},{1},{},{}}])
  input2_x = gradient_x:forward(input[{{},{2},{},{}}])

  input1_x = gradient_x:forward(input1_x)
  input2_x = gradient_y:forward(input2_x)

  gradient_y = nn.Sequential()
  gradient_y:add(nn.SpatialConvolution(1,1,3,3,1,1,1,1))
  weights = gradient_y:get(1).weight
  weights = torch.Tensor{{1,1,1},{0,0,0},{-1,-1,-1}}

  gradient_y:cuda()

  input1_y = gradient_y:forward(input[{{},{1},{},{}}])
  input2_y = gradient_y:forward(input[{{},{2},{},{}}])

  input1_y = gradient_x:forward(input1_y)
  input2_y = gradient_y:forward(input2_y)

  reg_I = torch.Tensor(input:size()):cuda()
  reg_J = torch.Tensor(input:size()):cuda()

  reg_I[{{},{1},{},{}}]:copy(input1_y)
  reg_I[{{},{2},{},{}}]:copy(input2_y)

  reg_J[{{},{1},{},{}}] = input1_x
  reg_J[{{},{1},{},{}}] = input2_x
  --print(reg_J)

  self.gradInput = self.gradInput + reg_I + reg_J

  return self.gradInput
end
