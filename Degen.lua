--[[
-- Say the numeral corresponding to 1 more than the current tokens/numeral
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
weight = torch.pow(numerals,-2)
rep = torch.tril(torch.ones(num_numer-1,num_numer-1)):cat(torch.zeros(num_numer-1))
--rep = torch.eye(num_numer-1):cat(torch.zeros(num_numer-1))
rep = torch.zeros(1,num_numer):cat(rep,1)
--rep =  rep:cat(torch.eye(num_numer))
rep =  rep:cat(numerals)
print(rep)

local last_said = nn.Identity()()
local hid = nn.ReLU()(nn.Linear(num_numer,hid_dim)(last_said))
local say = nn.LogSoftMax()(nn.Linear(hid_dim,num_numer)(hid))
network = nn.gModule({last_said},{say})
w,dw = network:getParameters()
crit = nn.ClassNLLCriterion()

train = function(w)
    network:zeroGradParameters()
    local samples = distributions.cat.rnd(mb_dim,weight,{categories=rep})
    data = samples[{{},{1,num_numer}}]
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
        gnuplot.imagesc(output)
        gnuplot.figure(2)
        _,best = output:max(2)
        best = best[{{},1}]:double()
        for n = 1,num_numer do
            if best[n] == n then
                if learn_time[n] == 0 then
                    learn_time[n] = i
                end
            else
                learn_time[n] = 0 
            end
        end
        --gnuplot.bar(learn_time)
        gnuplot.plot(best)
        sys.sleep(2)
        cumloss = 0
        if (best-numerals):nonzero():dim() == 0 then
            break
        end
    end
end



