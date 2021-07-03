function genfig2D(data, strAlg, strInst)

% 创建 figure
figure1 = figure;

% 创建 axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

Markers = {
    'o',...
    '>',...
    '<'...
    'v',...
    '^',...
    'diamond',...
    'pentagram',...
    'square',...
    'hexagram',...
    '*',...
    '+',...
    'x',...
    '>',...
    '<',...
    'v',...
};
Markersizes = {
    6,...
    6,...
    6,...
    6,...
    6,...
    6,...
    6,...
    6,...
    6,...
    6,...
    12,...
    12,...
    12,...
    12,...
    12,...
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
};

%%
tarINDs = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15];
lenINDs = length(tarINDs);

%%
count = 1;
for iAlg = tarINDs
    for i = 1
%         tmp = data{iAlg,i,1};
%         scatter(tmp(:,1),tmp(:,2),'Marker',...
%             Markers{count},'MarkerEdgeColor',colors{count},'MarkerFaceColor',colors{count},...
%             'DisplayName',strrep(strAlg{iAlg,i,1},'_','\_'));
        tmp = data{iAlg,i,2};
        scatter(tmp(:,1),tmp(:,2),'Marker',...
            Markers{count},'MarkerSize',Markersizes{count},'MarkerEdgeColor',colors{count},...
            'DisplayName',strrep(strAlg{iAlg,i,2},'_','\_'));
        count = count + 1;
        %legendName{iAlg,i,1} = sprintf('%s-%s-TRAIN', algs{iAlg}, models{i});
        %legendName{iAlg,i,2} = sprintf('%s-%s-TEST',  algs{iAlg}, models{i});
    end
end

% 创建 title
tmp=sprintf('%s',strInst);
title(tmp);

% 创建 xlabel
xlabel('f_{ACE}');
% 创建 ylabel
ylabel('f_{MCE}');

% % 取消以下行的注释以保留坐标轴的 X 范围
% xlim(axes1,[0 ceil(1.5*max(X1))]);
% % 取消以下行的注释以保留坐标轴的 Y 范围
% ylim(axes1,[0 ceil(1.5*max(Y1))]);
grid(axes1,'on');
% 设置其余坐标轴属性
set(axes1,'FontSize',11,'FontWeight','bold');
% 创建 legend
%legend(axes1,'show');
legend(axes1,'show','Location','best');

