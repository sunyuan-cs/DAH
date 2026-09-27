function [cellX, label] = sample(name, sele_num)
load (name);
%sele_num = 10;
fea = fea';
nnClass = length(unique(gnd));                         
num_Class=[];
for i = 1:nnClass
    num_Class = [num_Class length(find(gnd==i))];      
end
s = 0;
label = [];
for j = 1:nnClass
    cellnum = floor(num_Class(j)/sele_num); 
    randIdx  = randperm(num_Class(j));
    idx=find(gnd==j);
    for k = 1:cellnum
        s = s +1;
        cellX{s} = fea(:,idx(randIdx((k-1)*sele_num+1:k*sele_num)));
        label(s) = j;
    end
end

