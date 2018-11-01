clear all; clc;

NUM_GPUS = 4;     % Set to number of GPUs in the the system
smoothen = false; % Flag to smoothen the figure 

% Open output of profile_gpus.sh and format to arrays of utilization
% percentage per GPU (stored in matrix `gpus`)
fid = fopen('gpu_utillization.log', 'rt');
raw = textscan(fid, '%s');
raw = raw{1};
nums = regexp(raw,'\d+(\.)?(\d+)?','match');
nums = str2double([nums{:}]);
gpus = zeros(NUM_GPUS,length(nums)/NUM_GPUS);
for i = 1:NUM_GPUS:length(nums)
    gpus(i) = nums(i);
    gpus(i+1) = nums(i+1);
    gpus(i+2) = nums(i+2);
    gpus(i+3) = nums(i+3);
end

% Smoothen the plots if preferred (will also reduce amplitudes)
if smoothen
    for i = 1:NUM_GPUS
       gpus(i,:) = smooth(gpus(i,:));
    end
end

generate_plot(gpus');
