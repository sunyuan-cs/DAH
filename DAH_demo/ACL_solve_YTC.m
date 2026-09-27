function result = ACL_solve_YTC(DATA,random_index,alpha,beta,lambda,nbits,anchor_num,frameNum)
% load random_kernel_YTC
options.KernelType = 'Gaussian';
% frameNum =50;
trainN = 47;
testN = 47;
dl=10;
trainSetNum=trainN*3;


for index=1:5
    
    X = [];
    Dic = [];
    tr_data=cell(1,trainSetNum); % 47个人，每个人3个cell用来训练
    ytrain=[];%标签
    for i=1:trainN
        for k=1:3
            temp1=DATA{i,random_index{index}(i,k)}; %随机选一个用于训练
            if frameNum > size(temp1,2)
                temp1=[temp1 temp1 temp1 temp1];
                if frameNum <= size(temp1,2)
                    temp1=temp1(:,1:frameNum);
                end
            else
                temp1=temp1(:,1:frameNum);         %选取一部分样本
            end
            temp1 = NormalizeFea(temp1, 0);
            ytrain=[ytrain,i*ones(1,size(temp1,2))];
            Dic=[Dic,temp1];
        end
    end
    
    %     Y_label=zeros(frameNum*trainN,trainN);
    %     for i_label = 1:size(ytrain,2)
    %         Y_label(i_label,ytrain(i_label))=1;
    %     end
    
    %costruct set label matrix and relation matrix
    Y_label=zeros(trainN,trainSetNum);
    for i_label = 1:trainSetNum
        Y_label(ytrain(i_label*frameNum),i_label)=1;
    end
    
    
    I_label=zeros(trainN,trainSetNum*frameNum);
    for i_label = 1:frameNum*trainSetNum
        I_label(ytrain(i_label),i_label)=1;
    end
    
    S_label=Y_label'*I_label;
    
%     options.t = mean(mean(EuDist2(Dic)));
%     anchor_choose=[];    
    % %fix anchor
    %     ran_index=ran_index(1,1:anchor_num);
    %     for inter = 1:trainSetNum
    %         anchor_choose = [anchor_choose,ran_index+(inter-1)*frameNum];
    %     end
    %     K = buildKernel(Dic',Dic(:,anchor_choose)',options)';
    
    
    %fix anchor
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
    
    len = nbits;
    [~, n] = size(S_label);
    
    
    for cross_valid=1:1
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
%             LapA=B;
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
            
%             LapAold=B;
%             obj(lenoops) = norm(LapA-LapAold,'fro')^2/norm(LapAold,'fro')^2;
            
        end
        %     plot(obj);
        B_train = B;
        
        
        
        s=0;
        for it=1:testN
            for k=4:9
                s=s+1;
                Y=DATA{it,random_index{index}(it,k)};
                if frameNum > size(Y,2)
                    Y=[Y Y Y Y];
                    if frameNum <= size(Y,2)
                        Y=Y(:,1:frameNum);
                    end
                else
                    Y=Y(:,1:frameNum);         %选取一部分样本
                end
                Y = NormalizeFea(Y, 0);
                %              K_Y = buildKernel(Y',Dic(:,anchor_choose)',options)';
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
                    ind_ham=ceil(col_min_value/(3*frameNum));%对应类别号
                    
                    for index_ham=1:size(ind_ham)
                        flag=ind_ham(index_ham);
                        label(flag)=label(flag)+1;
                    end
                    [~,index_vote1] = max(label);
                    label_each(index_vote1) = label_each(index_vote1)+1;
                end
                [~,index_vote] = max(label_each);
                
                
                
                %show results
                %             if it ~= index_vote
                %                 fprintf('[%2d]True:%2d, Pred:%d\tX\n',s,it, index_vote);
                %             else
                %                 fprintf('[%2d]True:%2d, Pred:%d\n',s,it, index_vote);
                %             end
                pred(s) = index_vote;
                groudtrue(s) = it;

                   end
        end
        Rate_cross(cross_valid)=length(find(pred==groudtrue))./length(groudtrue);
    end
    ClassRate(index) = mean(Rate_cross);
    fprintf('[%d][%d] #ACC= %.2f \n',index,frameNum, ClassRate(index)*100);
    
    
end
toc
acuu_mean=mean(ClassRate);
acc_std=std(ClassRate);
result=[acuu_mean acc_std]
end
