function result = ACL_solve_ETH(ETH_new,random_index_ETH80,alpha,beta,lambda,nbits,anchor_num,frameNum)
%set_and_instance
totalSetNumPer=10;%每个类别对应的集合总数10
trainSetNumPer=5;%每个类别对应的训练集数5
trainN = 8;%训练集有8个类别
testN=8;%测试集8个类别
testSetNumPer= totalSetNumPer-trainSetNumPer;%%每个类别对应的测试集数5
trainSetNum=trainN*trainSetNumPer;%训练集数40
testSetNum= trainN*testSetNumPer ;%测试集数40
Xtrain=cell(trainSetNum,1);
Xtest=cell(testSetNum,1);
ytest=zeros(testSetNum,1);

ClassRate=[];
options.KernelType = 'Gaussian';
dl=30;
load random_kernel_ETH

for index=1:5  
    %For training
    trainS=0;
    Dic=[];
    X = [];
    ytrain=[];
    for i=1:trainN
        for k=1:trainSetNumPer
            trainS=trainS+1;
            temp1=double(ETH_new{i,random_index_ETH80{index}(i,k)}(:,1:frameNum));                
          if frameNum > size(temp1,2)
                temp1=temp1(:,1:size(temp1,2));
            else
                temp1=temp1(:,1:frameNum);         %选取一部分样本
          end
            temp1 = NormalizeFea(temp1, 0);
            X = [X,temp1];
            Xtrain{trainS,1}=temp1;
            ytrain=[ytrain,i*ones(1,size(temp1,2))];
            temp1 = NormalizeFea(temp1, 0);
            Dic = [Dic,temp1];
        end
    end      
    
    %costruct set label matrix and relation matrix 
    Y_label=zeros(trainN,trainSetNum);
    for i_label = 1:trainSetNum
        Y_label(ytrain(i_label*frameNum),i_label)=1;
    end
       
    
    I_label=zeros(trainN,frameNum*trainSetNum);
    for i_label = 1:frameNum*trainSetNum
        I_label(ytrain(i_label),i_label)=1;
    end
    
    S_label=Y_label'*I_label;
    
   
    
%fix anchor 
    options.t = mean(mean(EuDist2(Dic)));
    anchor_choose=[];
    ran_index=ran_index(1,1:anchor_num);
    for inter = 1:trainSetNum
        anchor_choose = [anchor_choose,ran_index+(inter-1)*frameNum];
    end    
    K = buildKernel(Dic',Dic(:,anchor_choose)',options)';
    
%      K=Dic;
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
        obj_ETH(lenoops) = norm(LapA-LapAold,'fro')^2/norm(LapAold,'fro')^2;
        
        
    end
    plot(obj_ETH);
    B_train = B;
    %-------------- test------------------------------
    testS=0;
    for it=1:testN
        for k=trainSetNumPer+1:totalSetNumPer
            testS=testS+1;
            temp2=double(ETH_new{it,random_index_ETH80{index}(it,k)}(:,1:frameNum));
            Y = NormalizeFea(temp2, 0);
%             if size(temp2,2)>parncol
%                 dtemp=dic_com(temp2,parncol);
%                 Y = NormalizeFea(dtemp, 0);
%             end

            
            K_Y = buildKernel(Y',Dic(:,anchor_choose)',options)';
%           K_Y=Y;
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
            label = zeros(1,trainN);
            index_temp=[];
            label_each=zeros(1,trainN);         
            for i = 1 : size(Y,2)    %each image
                dis_col =  hammTrainTest(:,i);
                col_min_value=find(dis_col==min(dis_col));
                ind_ham=ceil(col_min_value/(5*frameNum));
                for index_ham=1:size(ind_ham)
                    flag=ind_ham(index_ham);
                    label(flag)=label(flag)+1;
                end
                [~,index_vote1] = max(label);
                label_each(index_vote1) = label_each(index_vote1)+1;
            end
            [~,index_vote] = max(label_each);
%             if it ~= index_vote
%                 fprintf('[%2d]True:%2d, Pred:%d\tX\n',testS,it, index_vote);
%             else
%                 fprintf('[%2d]True:%2d, Pred:%d\n',testS,it, index_vote);
%             end
            pred(testS) = index_vote;
            groudtrue(testS) = it;
        end
    end
    
    
    Rate_cross(cross_valid)=length(find(pred==groudtrue))./length(groudtrue);
    end
    ClassRate(index)=mean(Rate_cross);
    clear Rate_cross;
    
%     ClassRate(index) = length(find(pred==groudtrue))./length(groudtrue);
    fprintf('[%d][%d] #ACC= %.2f \n',index,frameNum, ClassRate(index)*100);


    
end
acuu_mean=mean(ClassRate);
acc_std=std(ClassRate);
result = [acuu_mean acc_std]
end


