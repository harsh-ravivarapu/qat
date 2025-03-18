function [Action] = create_stim(len,b,freq)
%create_stim generates stim for for step function use
%   Uses the general conversion that RL uses to generate a stim vector
%   Used in testing different regular stim performance

% so first row (or first len spaces) is for 0
% len is length of sim
% b=freq*len/1000

if freq==0
Action = zeros([len*b,1]);
else
% so we want 1s to occur spaced out evenly (steps/pulse)
% calculated by 1/(freq*stepsize)
step_size = 1/1000; % sec
spacing = floor(1/(step_size*freq)); % space we want between pulses in final group
% at step 4 we want a 1 occuring every 20 spaces (1,21,41..len+1-20)
% so at step 3 we want those indices
% which are extracted from the index of the maximum of each column of the
% action
% the action is shaped as lxb
% MATLAB treats A as if its elements were strung out in a long column 
% vector, by going down the columns consecutively
Action = zeros([len,b]);
offset=1;

for i=1:b
    Action(offset,i)=100;
    offset=offset+spacing;
end

Action = reshape(Action,[len*b,1]);
end
end

% Action = reshape(Action,[len b]);
%     Action = softmax(Action);
%     [~,idx] = max(Action,[],1);
%     pprofile = zeros(len,1);
%     pprofile(idx) = 1;
%     tmax = len;