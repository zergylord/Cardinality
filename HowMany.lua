require 'nngraph'
require 'optim'
require 'gnuplot'
num_numer = 10
local last_said = nn.Identity()()
local tokens = nn.Identity()()
local hid = nn.ReLU()(nn.CAddTable(){nn.Linear(num_numer,hid_dim)(last_said),nn.Linear(num_numer,hid_dim)(tokens)})
local say = nn.L
