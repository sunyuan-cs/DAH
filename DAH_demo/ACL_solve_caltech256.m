function result = ACL_solve_caltech256(DATA,label,alpha,beta,lambda,nbits,anchor_num,frameNum)


% SumSetNum=length(label);
SetNum = length(unique(label));
X = [];
Dic = [];
ytrain=[];%标签
groudtrue = [];
testNums = 0;
trainN=102;
% for index=1:10

for i=1:SetNum
    idx=find(label==i);
 
    temp1=double(DATA{idx(1)}); %随机选一个用于训练
    temp1 = NormalizeFea(temp1, 0);
    Dic=[Dic,temp1];

    for j=2:length(idx)
        testNums = testNums +1;
        temp1 =double(DATA{idx(j)});
        TestData{testNums} = NormalizeFea(temp1,0);
        groudtrue(testNums) = i;
    end
    
end

trainSetNum=SetNum;



Y_label=eye(SetNum);
I_label=kron(Y_label,ones(1,frameNum));
S_label=Y_label'*I_label;


%      ran_index=randperm(10,10);
    ran_index=[7,6,4,10,1,5,2,9,3,8];
    options.t = mean(mean(EuDist2(Dic)));
    anchor_choose=[];
    ran_index=ran_index(1,1:anchor_num);
    for inter = 1:trainSetNum
        anchor_choose = [anchor_choose,ran_index+(inter-1)*frameNum];
    end    
    K = buildKernel(Dic',Dic(:,anchor_choose)',options)';
% K=Dic;


len = nbits;
[~, n] = size(S_label);
dl=10;

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
        obj_CAL(lenoops) = norm(LapA-LapAold,'fro')^2/norm(LapAold,'fro')^2;
    end
    plot(obj_CAL);
    
    B_train = B;
    
    
    testS=0;
    for it=1:testNums
        testS=testS+1;
        Y=TestData{it};
        Y = NormalizeFea(Y, 0);
        
        
        
        K_Y = buildKernel(Y',Dic(:,anchor_choose)',options)';
%         K_Y=Y;
        B_test = sgn(W'*K_Y);
        %--------------hamming distance--------------------
        B_train_c = logical(B_train == 1);
        B_test_c = logical(B_test == 1);
        %----compact
        B = compactbit(B_train_c');
        tB = compactbit(B_test_c');
        %----hamming distance
        hammTrainTest = hammingDist(tB, B)';
        %------------classification--------------------
        label_pr = zeros(1,trainN);
        index_temp=[];
        label_each=zeros(1,trainN);
        for i = 1 : size(Y,2)    %each image
            dis_col =  hammTrainTest(:,i);
            col_min_value=find(dis_col==min(dis_col));
            ind_ham=ceil(col_min_value/(frameNum));
            for index_ham=1:size(ind_ham)
                flag=ind_ham(index_ham);
                label_pr(flag)=label_pr(flag)+1;
            end
            [~,index_vote1] = max(label_pr);
            label_each(index_vote1) = label_each(index_vote1)+1;
        end
        [~,index_vote] = max(label_each);
        %             if it ~= index_vote
        %                 fprintf('[%2d]True:%2d, Pred:%d\tX\n',testS,it, index_vote);
        %             else
        %                 fprintf('[%2d]True:%2d, Pred:%d\n',testS,it, index_vote);
        %             end
        pred(testS) = index_vote;
%         groudtrue(testS) = it;
%         groudtrue(testS)=groudlabel;

    end
Rate_cross(cross_valid)=length(find(pred==groudtrue))./length(groudtrue);
% fprintf('[%d][%d] #ACC= %.2f \n',cross_valid,frameNum, Rate_cross*100);
end
ClassRate=mean(Rate_cross)

acc_std=std(Rate_cross);
result = [ClassRate acc_std];

end





