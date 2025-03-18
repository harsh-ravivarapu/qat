function [avg_freq] = find_freq(stim,dt)
%find_freq finds the average frequency of a stimulus vector, given a step
%dt (ms)

% assume dt is in ms.
dt_sec = dt/1000;
amt_stims = 0;
for i = 2:length(stim)
    if (stim(i-1) > 0 && abs(stim(i))<0.0001) || (i==length(stim) && stim(i)>0)
        % if it was on last step but off this step, there was a stim pulse
        % or if we are at the last step and it is on, that is a stim pulse
        amt_stims = amt_stims + 1;
    end
end
time = length(stim)*dt_sec;
avg_freq = amt_stims/time;
end