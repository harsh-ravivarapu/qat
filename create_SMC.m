function [SMC]=create_SMC(frequency,len,dt,PW,base_amplitude,amplitude)
% function Idbs=creatdbs(tmax,dt,PW,amplitude)

% Define time vector
pulse_w = PW/dt; % Pulse width in index

% Create initial DBS array and pulse array (iD is the amplitude)
SMC=zeros(1,len);
iD=amplitude-base_amplitude;            % nA/cm2 [DBS current amplitude]
pulse=iD*ones(1,pulse_w);

% Loop through Idbs array and embed pulse array at random intervals while
% keeping the one second average frequence equal to the defined parameter
% above.

% variances
% want to implement more spaced out
% done with assignment

impulse = zeros(1,frequency); % list of impulse locations.
inst = frequency;
isi = 1000/inst;
ipi=round(isi*1/dt);
i=1;
n_im = 1;
while i<len
    j=randi([i+floor(ipi.*0.5) i+ipi],1,1); % j=randi([i i+ipi],1,1);
    if j > len
        break
    end
    SMC(j:j+pulse_w-1)=pulse;
    i=i+round(isi*1/dt);
    impulse(n_im)=j;
    n_im = n_im+1;
end
% replace 0s with base_amplitude
SMC = SMC + base_amplitude;
end