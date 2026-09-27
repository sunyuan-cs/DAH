%The algorithm is implemented by Matlab R2016b on a MAC platform with Intel Core i7 3.2GHz CPU and 16GB memory.
clear memory;
clear all;
addpath(genpath('database'))
addpath(genpath('utilize'))
addpath(genpath('varying_frames_ETH80'))
addpath(genpath('ksvdbox'))

load ETH80_10_10_41
load random_index_ETH80
% frameNum=15;

candidate_frameNum=[25,41];
candidate_nbits=[128,256];
candidate_alpha =[1e-4,1e-3,1e-2,1e-1];
candidate_beta=[1e-4,1e-3,1e-2,1e-1];
candidate_lambda=[1e-4,1e-3,1e-2,1e-1];
candidate_anchor_num=[3];



total_res = [];
for c_frameNum = 1: length(candidate_frameNum)
for c_nbits = 1: length(candidate_nbits)
    for c_alpha = 1: length(candidate_alpha)
        for c_beta = 1: length(candidate_beta)
            for c_lambda = 1: length(candidate_lambda)
                for c_anchor_num=1:length(candidate_anchor_num)
                    paras.frameNum = candidate_frameNum(c_frameNum);
                    paras.nbits = candidate_nbits(c_nbits);
                    paras.alpha = candidate_alpha(c_alpha);                                   
                    paras.beta = candidate_beta(c_beta);
                    paras.lambda = candidate_lambda(c_lambda);
                    paras.anchor_num = candidate_anchor_num(c_anchor_num)
                    tic                  
                    Out_result = ACL_solve_ETH(ETH_new,random_index_ETH80,paras.alpha,paras.beta,paras.lambda,paras.nbits,paras.anchor_num,paras.frameNum);                  
                    t=toc;
                    total_res = [total_res;Out_result, paras.alpha, paras.beta,paras.lambda, paras.nbits,paras.anchor_num,paras.frameNum,t];
                 fprintf('*');
                end
                fprintf('\n');
            end
        end
    end
end
end
storge_file = fullfile( ['acc_',  'ACL_ETH1.mat']);
save(storge_file, 'total_res');
