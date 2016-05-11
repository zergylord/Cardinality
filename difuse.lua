require 'gnuplot'
v = torch.zeros(100)
s = 50
for i=1,1e4 do
    v[s] = v[s] + 1
    if torch.rand(1)[1] > .5 then
        s = math.max(1,s-1)
    else
        s = math.min(100,s+1)
    end
end
gnuplot.bar(v)
