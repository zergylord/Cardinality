--[[
-- How many task without the motor part
-- {tokens,last said}->{say}
-- should say next numeral unless 0 tokens remain
--]]
require 'nngraph'
require 'optim'
require 'gnuplot'
require 'distributions'
num_numer = 20
hid_dim = 100
mb_dim = 32
num_steps = 1e4
refresh = 1e1
config = {learningRate=1e-3}


numerals = torch.range(1,num_numer)
num_unique = torch.range(1,num_numer+1):sum()-1
numer_weight = torch.pow(torch.range(1,num_numer),-2)
--numer_weight = torch.ones(num_numer)
weight = torch.zeros(num_unique)
rep = torch.zeros(num_unique,num_numer*2+2)
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
use_count = 1
ratio = torch.ones(num_steps)
anneal_end = 1e4
ratio[{{1,anneal_end}}] = torch.linspace(-6,6,anneal_end):sigmoid()
if use_count then --interleaved
    crep = torch.zeros(num_numer,num_numer*2+2)
    cweight = torch.ones(num_numer):mul(weight:sum()/num_numer)
    for prev=0,num_numer-1 do
        if prev > 0 then
            crep[prev+1][num_numer+prev] = 1
        end
        crep[prev+1][-2] = 1
        crep[prev+1][-1] = prev+1
    end
    rep = rep:cat(crep,1)
    weight = weight:cat(cweight)
end
gnuplot.figure(2)
gnuplot.bar(weight)
--gnuplot.imagesc(rep)

local input = nn.Identity()()
local hid1 = nn.ReLU()(nn.Linear(rep:size(2)-1,hid_dim)(input))
local hid = hid1
num_layers = 1
for i = 1,num_layers do
    hid = nn.ReLU()(nn.Linear(hid_dim,hid_dim)(hid))
end
local last_hid = hid
local say = nn.LogSoftMax()(nn.Linear(hid_dim,num_numer+1)(last_hid))
network = nn.gModule({input},{say})
w,dw = network:getParameters()
crit = nn.ClassNLLCriterion()

train = function(w)
    network:zeroGradParameters()
    local samples = distributions.cat.rnd(mb_dim,cur_weight,{categories=rep})
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
for t = 1,num_steps do
    cur_weight = weight:clone()
    cur_weight[{{1,num_unique}}]:mul(ratio[t])
    cur_weight[{{num_unique+1,-1}}]:mul(1-ratio[t])
    _,batchloss = optim.adam(train,w,config)
    cumloss = cumloss  + batchloss[1]
    if t % refresh == 0 then
        print(t,cumloss,w:norm(),dw:norm())
        output = network:forward(rep[{{1,num_unique},{1,-2}}])
        
        gnuplot.figure(1)
        gnuplot.imagesc(torch.exp(output))

        gnuplot.figure(2)
        local temp =rep[{{1,num_unique},-1}]:reshape(num_unique,1):long()
        good = torch.exp(output:gather(2,temp))
        --gnuplot.plot(id,good,'+')
        --gnuplot.axis{'','',0,1}
        percent_perfect = torch.ones(num_numer)
        for i=1,num_unique do
            percent_perfect[id[i] ] = percent_perfect[id[i] ] * good[i][1]
        end
        gnuplot.bar(percent_perfect)
        gnuplot.axis{.5,.5+num_numer,0,1}

        gnuplot.figure(3)
        done = true
        for i=1,num_numer do
            if learn_time[i] == 0 then
                done = false
                if percent_perfect[i] > .8 then
                    learn_time[i] = t
                end
            end
        end
        gnuplot.plot(learn_time)

        --sys.sleep(1)
        cumloss = 0
        if done then
            print(t)
            break
        end
    end
end



