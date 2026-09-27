function D=dic_com(DT,parncol,method)
%dictionary learning (DT)

method='ksvd';
if strcmp(method,'metaface')
%==================learn the dictionary on the training data=======
par.lambda_l     =    0.001;          % parameter of l1_ls in learning
par.lambda_t     =    0.001;          % parameter of l1_ls in testing
par.objT         =    1e-2;           % the objective gap of metaface learning
par.nIter        =    25;             % the maximal iteration number of metaface learnng
par.ncol         =    parncol;
DT       =    DT./ repmat(sqrt(sum(DT.*DT)),[size(DT,1) 1]); % unit norm 2
[D,alpha]=Metaface_rand(DT,par.ncol,par.lambda_l,par.objT,par.nIter);%Metaface Dictionary Learning
elseif strcmp(method,'ksvd') 
DT       =    DT./ repmat(sqrt(sum(DT.*DT)),[size(DT,1) 1]); % unit norm 2    
params.data = DT;
params.Tdata = 3;
params.dictsize = parncol;
params.iternum = 30;
params.memusage = 'high';
D= ksvd(params,'');

elseif strcmp(method,'spam') 
  param.K=parncol;  
  param.lambda=0.001;
  param.iter=30;
  [D]=mexTrainDL(DT,param);

end