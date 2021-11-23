function [S_all] = fun_read_data(fnm)
%READ_DATA 此处显示有关此函数的摘要
%   此处显示详细说明
S_all = {};

fid=fopen(fnm);
i = 1;
while ~feof(fid)
   tline = fgetl(fid);
   if isempty(tline)
       continue;
   end
   S = regexp(tline, ',+', 'split');
   tmp = [];
   for j = 1:length(S)
       if isempty(S{j})
           continue;
       end
       tmp = [tmp str2num(S{j})];
   end
   S_all{i} = tmp;
   i = i + 1;
end       

end

