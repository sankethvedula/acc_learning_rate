require "nn"
require "MSEUpdatedCriteria"

Criterion = nn.MSEUpdatedCriteria()

input = torch.rand(10,2,28,28)
target = torch.rand(10,2,28,28)

-- gradient_y = nn.Sequential()
-- gradient_y:add(nn.SpatialConvolution(1,1,3,3,1,1,1,1))
-- weights = gradient_y:get(1).weight
-- weights = {{1,1,1},{0,0,0},{-1,-1,-1}}
--
--
-- gradient_x = nn.Sequential()
-- gradient_x:add(nn.SpatialConvolution(1,1,3,3,1,1,1,1))
-- weights = gradient_x:get(1).weight
-- weights = {{1,0,-1},{1,0,-1},{1,0,-1}}
--
-- print(gradient_x)
-- print(gradient_y)

--out = gradient_x:forward(input)
--print(out:size())

loss = Criterion:forward(input, target)
grad = Criterion:backward(input,target)
