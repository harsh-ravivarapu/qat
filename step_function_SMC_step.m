function [Observation,Reward,isdone,IT] = step_function_SMC_step(Action,IT,freq,len,b,dt,stride,windowlen)
% Step function
    %% Action Logits to Action
    % change reshape to ensure it can be flexible (changed)
    %reshape_len = length(Action)/b;
    %Action = reshape(Action,[reshape_len b]); % changed from len
    %Action = softmax(Action);
    %[~,idx] = max(Action,[],1);
    %pprofile = zeros(reshape_len,1); % changed from len
    %pprofile(idx) = 1;
    pprofile = Action; % added when removing Action Logit to Action Process
    tmax = len;

    %% Simulation
    [beta_vec, EI, ~] = rl_BGM_step_SMC_pulse_python_step(tmax,IT,1,0,freq,pprofile,len,dt,stride,windowlen); %(tmax,IT,pd,corstim,freq,pprofile,len,CTX_workspace,dt,stride)

    %% Load info
    % Scale beta to be [0,1]
    beta_vec = beta_vec./1000;

    % Log state and other variables needed
     LoggedSignal.State = [beta_vec]; %
%    LoggedSignal.State = [beta_vec;EI];
    IT = IT+1;
    Observation = LoggedSignal.State;

    % Visualize Beta
    %if IT == 25
    %    figure;
    %    plot((1:length(LoggedSignal.All))/1000,LoggedSignal.All);
    %    hold on;
    %    xlabel("Seconds");
    %    ylabel("Beta_-spec Power");
    %    title(append("Beta Vector"));
    %    ylim([150 650])
    %    hold off;
    %    t = datetime('now',Format='uuuuMMdd''T''HHmmss');
    %    DateString = string(t);
    %    name = append('beta, ',DateString,'.png');
    %    saveas(gcf,name);
    %end
    

    %% START REWARD FUNCTION
    %% Reward Function Edited (number 7)
    isdone = false; % Done flag set to true if beta is too large
%    Reward = 2000*(1/100^2)^mean(beta_vec);
    beta_threshold = 0.35;
    if (mean(beta_vec) < beta_threshold)
        Reward = ((mean(beta_vec) - 0.35) *10)^2;
    else
        Reward = - ((mean(beta_vec) - 0.35) *10)^2;
    end
%    if (mean(beta_vec) < beta_threshold)
%        Reward = 1;
%    elseif (mean(beta_vec) < beta_threshold * 2)
%        Reward = 0;
%    else
%        Reward = -1;
%    end
%    if (mean(beta_vec) < beta_threshold) && (mean(EI) < 0.1)
%        Reward = 100;
%%        isdone = true;
%    elseif (mean(beta_vec) < beta_threshold) && (mean(EI) >= 0.1)
%        Reward = 1;
%%    elseif (mean(beta_vec) < 0.5) && (mean(EI) >= 0.1)
%%        Reward = 0.5;
%%    elseif (mean(beta_vec) < 0.75) && (mean(EI) >= 0.1)
%%        Reward = 0.25;
%    elseif (mean(beta_vec) >= beta_threshold) && (mean(EI) < 0.1)
%        Reward = 1;
%    else
%        Reward = -1;
%        if (mean(beta_vec)) >= 2.0
%            isdone = true;
%        end
%    end
    
    % if mean(beta_vec) > 650./1000 & mean(EI)
	% 	isdone = true;
	% 	Reward = -100;
	% elseif mean(beta_vec) > 500./650
	% 	Reward = -10;
	% elseif mean(beta_vec) > 400./650
	% 	Reward = -1;
	% elseif mean(beta_vec) > 350./650
	% 	Reward = 1;
	% elseif mean(beta_vec) > 325./650
	% 	Reward = 10;
	% else
	% 	Reward = 100;
	% end

    %% END REWARD FUNCTION
end
