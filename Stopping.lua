--[[
-- How many task without the motor part
-- {tokens,last said}->{say}
-- should say next numeral unless 0 tokens remain
--]]
require 'nngraph'
require 'optim'
require 'gnuplot'
require 'distributions'
num_numer = 10
hid_dim = 100
mb_dim = 32
num_steps = 1e4
refresh = 1e1
config = {learningRate=1e-3}


numerals = torch.range(1,num_numer)
num_unique = torch.range(1,num_numer+1):sum()
--weight = torch.pow(num_unique,-2)
weight = torch.ones(num_unique)
rep = torch.zeros(num_unique,num_numer*2+1)
ind = 1
for tok=0,num_numer do
    for prev=0,num_numer-tok do
        if prev > 0 then
            --last said
            rep[ind][num_numer+prev] = 1
        end
        if tok > 0 then
            --tokens
            rep[ind][{{1,tok}}] = 1
            --answer
            rep[ind][-1] = prev+1
        else
            rep[ind][-1] = num_numer+1 --done
        end
        ind = ind + 1
    end
end
gnuplot.imagesc(rep)

local input = nn.Identity()()
local hid = nn.ReLU()(nn.Linear(rep:size(2)-1,hid_dim)(input))
local say = nn.LogSoftMax()(nn.Linear(hid_dim,num_numer+1)(hid))
network = nn.gModule({input},{say})
w,dw = network:getParameters()
crit = nn.ClassNLLCriterion()

train = function(w)
    network:zeroGradParameters()
    local samples = distributions.cat.rnd(mb_dim,weight,{categories=rep})
    data = samples[{{},{1,-2}}]
    --target = samples[{{},{num_numer+1,-1}}]
    target = samples[{{},-1}]
    output = network:forward(data)
    loss = crit:forward(output,target)
    grad = crit:backward(output,target)
    network:backward(data,grad)
    return loss,dw
end
cumloss = 0
learn_time = torch.zeros(num_numer)
for i = 1,num_steps do
    _,batchloss = optim.adam(train,w,config)
    cumloss = cumloss  + batchloss[1]
    if i % refresh == 0 then
        print(i,cumloss,w:norm(),dw:norm())
        output = network:forward(rep[{{},{1,-2}}])
        gnuplot.figure(1)
        cumloss = 0
        gnuplot.imagesc(output:exp())
    end
end



