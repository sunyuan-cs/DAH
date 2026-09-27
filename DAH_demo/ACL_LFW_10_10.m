clear;
clc;

%author: Yuan Sun
%department: Sichuan university

warning off
clear all;
close all;
warning off
load LFW_10x10
%load LFW_VggFace2_Resnet50_2048
frameNum=10;
[DATA, label]= sample('LFW_10x10',frameNum);
%[DATA, label]= sample('LFW_VggFace2_Resnet50_2048',frameNum);

candidate_nbits=[128];%[16,32,64,128];
candidate_alpha =[1e-4,1e-3,1e-2,1e-1,1,10,100];
candidate_beta=[1e-4,1e-3,1e-2,1e-1,1,10,100];
candidate_lambda=[1e-4];
candidate_anchor_num=[3];
total_res = [];
for c_nbits = 1: length(candidate_nbits)
    for c_alpha = 1: length(candidate_alpha)
        for c_beta = 1: length(candidate_beta)
            for c_lambda = 1: length(candidate_lambda)
                for c_anchor_num=1:length(candidate_anchor_num)
                    paras.nbits = candidate_nbits(c_nbits);
                    paras.alpha = candidate_alpha(c_alpha);
                    paras.beta = candidate_beta(c_beta);
                    paras.lambda = candidate_lambda(c_lambda);
                    paras.anchor_num = candidate_anchor_num(c_anchor_num)
                    tic
                    Out_result = ACL_solve_LFW(DATA,label,paras.alpha,paras.beta,paras.lambda,paras.nbits,paras.anchor_num,frameNum);
                    t=toc;
                    total_res = [total_res;Out_result, paras.alpha, paras.beta,paras.lambda, paras.nbits,paras.anchor_num,t];
                    fprintf('*');
                end
                fprintf('\n');
            end
        end
    end
end

storge_file = fullfile( ['acc_',  'ACL_LFW_10_10.mat']);
save(storge_file, 'total_res');
