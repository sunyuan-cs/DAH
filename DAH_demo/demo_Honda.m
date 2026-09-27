clear memory;
clear all;
addpath(genpath('database'))
addpath(genpath('ksvdbox'))
warning off
load HONDA_data.mat

TestLab = testlabels;
TrainLab = traininglabels;
TrainData = TrainingSets;
TestData = TestSets;


options.KernelType = 'Gaussian';
% options.t = 1;

frameNum =50;
dl=10;
trainSetNum=20;


total_res=[];
ClassRata_data=[];

%生成训练数据
trainN = length(unique(TrainLab));
for nbits=[128]
for alpha =[0]
for beta=[1e-5,1e-4,1e-3,1e-2,1e-1,1,10]
for lambda=[1e-5,1e-4,1e-3,1e-2,1e-1,1,10]
for anchor_num=[5]
    
    
ytrain= [];
Dic = [];



    
    
    
    
for i = 1:trainN
    temp1 = double(TrainingSets(:,find(TrainLab==i)));
    if size(temp1,2)>=frameNum
        temp1 = temp1(:,1:frameNum);
    end
 % temp1 = z_avg(temp1, 1);   %0均值
   temp1 = NormalizeFea(temp1, 0);
    
    ygallery(i) = i;
    ytrain=[ytrain,i*ones(1,size(temp1,2))];
    Dic=[Dic,temp1];
end
 Y_label=zeros(trainN,trainSetNum);
    for i_label = 1:trainSetNum
        Y_label(ytrain(i_label*frameNum),i_label)=1;
    end
    
    
    I_label=zeros(trainN,trainSetNum*frameNum);
    for i_label = 1:frameNum*trainSetNum
        I_label(ytrain(i_label),i_label)=1;
    end
    
    S_label=Y_label'*I_label;



% Y_label=zeros(frameNum*trainN,trainN);
% for i_label = 1:size(ytrain,2)
%     Y_label(i_label,ytrain(i_label))=1;
% end
% 
% ran_index = randperm(frameNum);%size(Dic,2)
% ran_index = ran_index(1:anchor_num);
% anchor_choose=[];
% for inter = 0:trainN-1
%     anchor_choose = [anchor_choose,ran_index+frameNum*inter];
% end
% K = buildKernel(Dic',Dic(:,anchor_choose)',options);

 if anchor_num==0
        K=Dic';
    else
        options.t = mean(mean(EuDist2(Dic)));
        anchor_choose=[];
        ran_index=randperm(frameNum,anchor_num);
        anchor_sort=sort(ran_index);
        for inter = 1:trainSetNum
            anchor_choose = [anchor_choose,anchor_sort+(inter-1)*frameNum];
        end
        K = buildKernel(Dic',Dic(:,anchor_choose)',options)';
    end

%生成测试数据
testNum = length(TestLab);
Xtest = cell(testNum,1);
j = 0;
ytest = zeros(testNum,1);
for i = 1: testNum
    j = j + 1;
    temp1 = double(TestData{1,i});
    if size(temp1,2)>=frameNum
        temp1 = temp1(:,1:frameNum);
    else
        temp1 = temp1(:,1:size(temp1,2));
    end
%     temp1 = z_avg(temp1, 1);   %0均值
     temp1 = NormalizeFea(temp1, 0);
    Xtest{j}=temp1;
    ytest(j)=TestLab(i);
    
%     K_Y{j} = buildKernel(Xtest{j}',Dic(:,anchor_choose)',options);
    
end


len = nbits;
[~, n] = size(S_label);
    
    
    for cross_valid=1:5
        %------------------------------initialize--------------------------------------------
        
        %-----inital H
        H = randsrc(len,trainSetNum,[-1,1]);
        
        %-----inital B
        B=kron(H,ones(1,frameNum));
        
        %-----inital W
        W=(K*K'+lambda*eye(size(K,1)))\(K*B');
        
        %-----inital P
        P=(alpha*Y_label*Y_label'+alpha*I_label*I_label'+lambda*eye(trainN))\(alpha*Y_label*H'+alpha*I_label*B');
        
        
        
        %------------------------------training--------------------------------------------
        
        for lenoops=1:dl
            LapA=B;
            %--------- H-step
            H_C = alpha*P'*Y_label+beta*len*B*S_label';
            H= hash_const(H_C,len,trainSetNum);
            %--------- B-step
            B_C = W'*K+alpha*P'*I_label+beta*len*H*S_label;
            B=hash_const(B_C,len,n);
            %--------- W-step
            W=(K*K'+lambda*eye(size(K,1)))\(K*B');
            %--------- P-step
            P=(alpha*Y_label*Y_label'+alpha*I_label*I_label'+lambda*eye(trainN))\(alpha*Y_label*H'+alpha*I_label*B');
            
            LapAold=B;
            obj(lenoops) = norm(LapA-LapAold,'fro')^2/norm(LapAold,'fro')^2;
            
        end
        %     plot(obj);
        B_train = B;
        
        
        
        %     toc

s = 0;

for it=1:testNum
     s = s + 1;
    Y=Xtest{it};

    
    if anchor_num==0
        K_Y=Y';
    else
        K_Y = buildKernel(Y',Dic(:,anchor_choose)',options)';
    end
    B_test = sgn(W'*K_Y);
    

    
    
           
   %--------------hamming distance--------------------
                B_train_c = logical(B_train == 1);
                B_test_c = logical(B_test == 1);
                %每一行压缩成一个点
                B = compactbit(B_train_c');
                tB = compactbit(B_test_c');
                %求海明距离
                hammTrainTest = hammingDist(tB, B)';
                
                %------------投票策略--------------------
                label = zeros(1,trainN);
                index_temp=[];
                label_each=zeros(1,trainN);
                for i = 1 : size(Y,2)    %each image
                    dis_col =  hammTrainTest(:,i);
                    col_min_value=find(dis_col==min(dis_col));%寻找列最小值
                    ind_ham=ceil(col_min_value/(frameNum));%对应类别号
                    
                    for index_ham=1:size(ind_ham)
                        flag=ind_ham(index_ham);
                        label(flag)=label(flag)+1;
                    end
                    [~,index_vote1] = max(label);
                    label_each(index_vote1) = label_each(index_vote1)+1;
                end
                [~,index_vote] = max(label_each);
    
    
       
    
       
%                 if ytest(it) ~= index_vote
%                     fprintf('[%2d]True:%2d, Pred:%d\tX\n',s,ytest(it), index_vote);
%                 else
%                     fprintf('[%2d]True:%2d, Pred:%d\n',s,ytest(it), index_vote);
%                 end
    pred(s) = index_vote;
    groudtrue(s) = ytest(it);
end
 Rate_cross(cross_valid)=length(find(pred==groudtrue))./length(groudtrue);

    end
acuu_mean=mean(Rate_cross);
acc_std=std(Rate_cross);
result=[acuu_mean acc_std]
total_res = [total_res;result, alpha,beta, lambda, nbits,frameNum];

end
end
end
end
end

