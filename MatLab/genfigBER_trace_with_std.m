function genfigBER_trace_with_std(iters, Data_mean, Data_std, strInst, legendName, tmpStr)

%  YMATRIX1:  errorbar y 矩阵数据
%  DMATRIX1:  errorbar delta 矩阵数据

% 创建 figure
figure1 = figure;

% 创建 axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

Markers = {
    '+'
    'o'
    '*'
    '.'
    'x'
    'square'
    'diamond'
    'v'
    '^'
    '>'
    '<'
    'pentagram'
    'hexagram'
    '+'
    'o'
    '*'
    '.'
    'x'
    'square'
    'diamond'
    'v'
    '^'
    '>'
    '<'
    'pentagram'
    'hexagram'
    '+'
    'o'
    '*'
    '.'
    'x'
    'square'
    'diamond'
    'v'
    '^'
    '>'
    '<'
    'pentagram'
    'hexagram'
};
Markersizes = {
    6
    6
    6
    6
    6
    6
    6
    6
    6
    6
    6
    6
    6
    12
    12
    12
    12
    12
    12
    12
    12
    12
    12
    12
    12
    12
    18
    18
    18
    18
    18
    18
    18
    18
    18
    18
    18
    18
    18
};
colors ={
    [     0    0.4470    0.7410];
    [0.8500    0.3250    0.0980];
    [0.9290    0.6940    0.1250];
    [0.4940    0.1840    0.5560];
    [0.4660    0.6740    0.1880];
    [0.3010    0.7450    0.9330];
    [0.6350    0.0780    0.1840];
    [0.0800    0.1700    0.5500];
    [1.0000    0.1700    0.5500];
    [0.0800    1.0000    0.5500];
    [0.0800    0.1700    1.0000];
    [0.8000    1.0000    0.0500];
    [1.0000    0.1700    1.0000];
    [0.0800    1.0000    1.0000];
    [0.0000    0.0000    0.0000];
    [     0    0.4470    0.7410];
    [0.8500    0.3250    0.0980];
    [0.9290    0.6940    0.1250];
    [0.4940    0.1840    0.5560];
    [0.4660    0.6740    0.1880];
    [0.3010    0.7450    0.9330];
    [0.6350    0.0780    0.1840];
    [0.0800    0.1700    0.5500];
    [1.0000    0.1700    0.5500];
    [0.0800    1.0000    0.5500];
    [0.0800    0.1700    1.0000];
    [0.8000    1.0000    0.0500];
    [1.0000    0.1700    1.0000];
    [0.0800    1.0000    1.0000];
    [0.0000    0.0000    0.0000];
};

%%
tarINDs = 1:size(Data_mean,1);% 11 12 13 14 15];
lenINDs = length(tarINDs);

%%
count = 1;
% 使用 errorbar 的矩阵输入创建多个误差条
% all_mean_over_trace = zeros(length(algs), length(modelName), 3, length(probs), nObj, NTRACE+1);
% all_min_over_trace = zeros(length(algs), length(modelName), 3, length(probs), nObj, NTRACE+1);
for i = tarINDs
    %
    tmp_mean = Data_mean(i,1,1,1,1,:);
    tmp_mean = tmp_mean(:);
    tmp_std  = Data_std(i,1,1,1,1,:);
    tmp_std  = tmp_std(:);
    errorbar(iters,tmp_mean, tmp_std,'LineWidth',1,...
            'DisplayName',strrep(legendName{count,1,1},'_','\_')...
            ,'Marker',Markers{count},'MarkerSize',Markersizes{count},'color',colors{count});
    %
    tmp_mean = Data_mean(i,1,2,1,1,:);
    tmp_mean = tmp_mean(:);
    tmp_std  = Data_std(i,1,2,1,1,:);
    tmp_std  = tmp_std(:);
    errorbar(iters,tmp_mean, tmp_std,'LineWidth',1,...
            'DisplayName',strrep(legendName{count,1,2},'_','\_')...
            ,'Marker',Markers{count},'MarkerSize',Markersizes{count},'color',colors{count},'LineStyle','--');
    %
    tmp_mean = Data_mean(i,1,3,1,1,:);
    tmp_mean = tmp_mean(:);
    tmp_std  = Data_std(i,1,3,1,1,:);
    tmp_std  = tmp_std(:);
    errorbar(iters,tmp_mean, tmp_std,'LineWidth',1,...
            'DisplayName',strrep(legendName{count,1,3},'_','\_')...
            ,'Marker',Markers{count},'MarkerSize',Markersizes{count},'color',colors{count},'LineStyle',':');
    %
%     tmp = Datas(i,1,4,1,1,:);
%     tmp = tmp(:);
%     plot(iters,tmp,'LineWidth',1,...
%             'DisplayName',strrep(legendName{count,1,4},'_','\_')...
%             ,'Marker',Markers{count},'color',colors{count},'LineStyle','-.');
    count = count + 1;
end

% 创建 xlabel
xlabel('FEs');

% 创建 title
tmp_mean=sprintf('%s', strInst);
title(tmp_mean);

% 创建 ylabel
ylabel(tmpStr);

% 取消以下行的注释以保留坐标轴的 X 范围
%xlim(axes1,[0 max(iters(1,:))]);
box(axes1,'on');

% 设置其余坐标轴属性
set(axes1,'FontSize',11,'FontWeight','bold','XGrid','on',...
    'YGrid','on','YMinorTick','on');

% 创建 legend
legend1 = legend(axes1,'show');
set(legend1,'Location','best');
