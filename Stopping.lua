--[[
-- How many task without the motor part
-- {tokens,last said}->{say}
-- should say next numeral unless 0 tokens remain
--]]
require 'nngraph'
require 'optim'
require 'gnuplot'
require 'distributions'
num_numer = 30
hid_dim = 1000
mb_dim = 32
num_steps = 1e4
refresh = 1e1
config = {learningRate=1e-3}


numerals = torch.range(1,num_numer)
num_unique = torch.range(1,num_numer+1):sum()-1
numer_weight = torch.pow(torch.range(1,num_numer),-2)
--numer_weight = torch.ones(num_numer)
weight = torch.zeros(num_unique)
rep = torch.zeros(num_unique,num_numer*2+1)
id = torch.zeros(num_unique) --which problem does each rep belong to?
ind = 1
for tok=0,num_numer do
    for prev=0,num_numer-tok do
        if prev ~= 0 or tok ~= 0 then -- dont do the zero problem
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
            id[ind] = prev+tok
            weight[ind] = numer_weight[id[ind] ]/(id[ind]+1) --chance of picking that task / number of steps in task
            ind = ind + 1
        end
    end
end
gnuplot.figure(2)
gnuplot.bar(weight)
--gnuplot.imagesc(rep)

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
        local temp =rep[{{},-1}]:reshape(num_unique,1):long()
        good = output:gather(2,temp):exp()
        gnuplot.figure(2)
        --gnuplot.plot(id,good,'+')
        --gnuplot.axis{'','',0,1}
        percent_perfect = torch.ones(num_numer)
        for i=1,num_unique do
            percent_perfect[id[i] ] = percent_perfect[id[i] ] * good[i][1]
        end
        gnuplot.bar(percent_perfect)
        gnuplot.axis{.5,.5+num_numer,0,1}

        gnuplot.figure(1)
        cumloss = 0
        gnuplot.imagesc(output:exp())
        --sys.sleep(1)
    end
end



