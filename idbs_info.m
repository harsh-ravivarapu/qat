function [avg_freq,ipi,pulsewidth,ipf] = idbs_info(idbs,dt)
%idbs_info This function takes in the idbs stim pattern and the time step
%in ms, and calculates relevant values
%   Any idbs pattern and the time step (in ms) for each input, and
%   calculates the total average frequency (Hz), the interpulse interval
%   (ipi) (ms), the pulse width (assuming constant) (ms) and the interpulse
%   frequency (Hz).


% idbs is the stim pattern
% dt is the time step, in ms
% avg_freq is the total average frequency, in Hz
% ipi is a list containing each interpulse interval, in ms
% pulsewidth is the width of the pulse, in ms
% ipf is the interpulse frequency, in Hz


% Calculate total average frequency
dt_sec = dt/1000;
amt_stims = 0;
for i = 2:length(idbs)
    if (idbs(i-1) > 0 && abs(idbs(i))<0.0001) || (i==length(idbs) && idbs(i)>0)
        % if it was on last step but off this step, there was a stim pulse
        % or if we are at the last step and it is on, that is a stim pulse
        amt_stims = amt_stims + 1;
    end
end
time = length(idbs)*dt_sec;
avg_freq = amt_stims/time;

% Calculate Pulse Width
unitsperpulse = 0;
for i = 2:length(idbs)
    if (i==2 && abs(idbs(1))>0.0001)
        % account for the fact that we skip the first spot
        unitsperpulse = unitsperpulse + 1;
    end
    if abs(idbs(i))>0.0001 
        % looking at a stim, increase counter
        unitsperpulse = unitsperpulse + 1;
    end
    if abs(idbs(i))<0.0001 && abs(idbs(i-1))>0.0001
        % we just finished looking at a stim, exit the loop
        break
    end
end
% calculate pulsewidth
pulsewidth = unitsperpulse*dt;


% calculate interpulse interval
ipi = [];
ipi_counter = -1;
for i = 2:length(idbs)
    if abs(idbs(i))<0.0001 && abs(idbs(i-1))>0.0001
        % last one was a stim, this one isn't, turn on counter
        ipi_counter = 1;
        continue
    elseif ipi_counter>0 && abs(idbs(i))<0.0001 && abs(idbs(i-1))<0.0001
        % last one is not stim, this one also isn't, update counter
        ipi_counter = ipi_counter + 1;
        continue
    elseif ipi_counter>0 && abs(idbs(i))>0.0001 && abs(idbs(i-1))<0.0001
        % this one is stim, last one was not, turn off counter
        ipi = [ipi ipi_counter*dt];
    end
end


% calculate interpulse frequency

% convert ipi from ms to sec
ipi_sec = ipi./1000;
ipf = 1./ipi_sec;

end