% This function is used to run tests regarding the reset and step function
% it can be used to find average performance of reward functions, test
% normal stimulation, graph beta and EI after stimulation is applied

%% Doing some runs (With Reward Function 6)
tic;
freq = 50;
l = 400;
steps = 2;
episodes = 1;
stride = 2; 
window_size = 200;
dt = 0.01;
b = (freq*l)/1000;

reward = [];

for i = 1:episodes
    [InitialObservation, IT] = reset_function_SMC_step(freq,l,dt,stride,window_size);
    this_reward = 0;
    for j = 1:steps
        Action = create_stim(l,b,freq);
        [Observation,Reward,isdone,IT] = step_function_SMC_step(Action,IT,freq,l,b,dt,stride,window_size);
        this_reward = this_reward + Reward;
    end
    reward = [reward this_reward];
end
toc

%% Loading last one
window = 1;

beta = [];
ei = [];

for i=1:steps
    load(append(int2str(i),"pd130rs.mat"))
    beta = [beta beta_vec];
    ei = [ei EI];
end

%% Saving Beta Power Data to CSV
x_beta = linspace(window_size, steps * l, length(beta)) * 0.001; % X-axis (time)
y_beta = beta; % Y-axis (PSD)
data_beta = [x_beta' y_beta']; % Combine into a matrix
writematrix(data_beta, 'beta_power_plot.csv'); % Save to CSV

%% Plot Beta Power
figure;
plot(x_beta, y_beta)
title(strcat('Power in Beta Frequency Band, GPi, Length ',int2str(l),'ms, ',int2str(freq),'Hz, Stride ',int2str(stride),'ms, Window Size ',int2str(window_size),'ms'))
xlabel('Time (sec)')
ylabel('PSD')
ylim([0 650])
savefig(strcat('Half Beta Length ',int2str(l),'ms ',int2str(freq),'Hz Stride ',int2str(stride),'ms Window Size ',int2str(window_size),'ms.fig'))
saveas(gcf,strcat('Half Beta Length ',int2str(l),'ms ',int2str(freq),'Hz Stride ',int2str(stride),'ms Window Size ',int2str(window_size),'ms.png'))

%% Observed Beta Power Data
load("0pd0rs.mat")
obs_beta = [beta_vec,beta];
obs_ei = [EI ei];
time = linspace(window_size, (steps+1)*l, length(obs_beta))*0.001;
len = l;

%% Saving Observed Beta Power Data to CSV
x_obs_beta = time; % X-axis (time)
y_obs_beta = obs_beta; % Y-axis (PSD)
data_obs_beta = [x_obs_beta' y_obs_beta']; % Combine into a matrix
writematrix(data_obs_beta, 'observed_beta_power_plot.csv'); % Save to CSV

%% Plot Observed Beta Power
figure;
plot(time, obs_beta)
hold on
plot([len/1000,len/1000],[0,1000])
title(strcat('Power in Beta Frequency Band, GPi, Length ',int2str(l),'ms, ',int2str(freq),'Hz, Stride ',int2str(stride),'ms, Window Size ',int2str(window_size),'ms'))
xlabel('Time (sec)')
ylabel('PSD')
ylim([0 650])
savefig(strcat('All Beta Length ',int2str(l),'ms ',int2str(freq),'Hz Stride ',int2str(stride),'ms Window Size ',int2str(window_size),'ms.fig'))
saveas(gcf,strcat('All Beta Length ',int2str(l),'ms ',int2str(freq),'Hz Stride ',int2str(stride),'ms Window Size ',int2str(window_size),'ms.png'))

%% Saving Reward Data to CSV
x_reward = 1:length(reward); % X-axis (Episodes)
y_reward = reward; % Y-axis (Reward)
data_reward = [x_reward' y_reward']; % Combine into a matrix
writematrix(data_reward, 'reward_plot.csv'); % Save to CSV
