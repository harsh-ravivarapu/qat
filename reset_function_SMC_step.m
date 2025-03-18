function [InitialObservation, IT] = reset_function_SMC_step(freq,len,dt,stride,windowlen)
% Reset Function
    % Create Necessary Variables for Run
    x = (len+windowlen)/len;
    tmax = x*len; % Amount of time sim is set to run, ms

    % Run environment and return biomarkers
    [beta_vec, EI, ~] = rl_BGM_reset_SMC_pulse_python_step(tmax,0,1,0,0,[],len,dt,stride,windowlen); % (tmax,IT,pd,corstim,freq,pprofile,len,dt,stride)
    
    % Rescale Beta to be in range [0,1]
    beta_vec = beta_vec./1000; 

    % Store Information
    LoggedSignal.State = [beta_vec]; %
%    LoggedSignal.State = [beta_vec;EI];
    InitialObservation = LoggedSignal.State;
    IT = 1;
end

