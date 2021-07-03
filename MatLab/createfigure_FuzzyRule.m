function createfigure_FuzzyRule(YMatrix1, tmp_tt)
%CREATEFIGURE(YMATRIX1)
%  YMATRIX1:  y 数据的矩阵

%  由 MATLAB 于 17-Jan-2021 17:16:28 自动生成

% 创建 figure
figure1 = figure;

% 创建 axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

% 使用 plot 的矩阵输入创建多行
plot1 = plot(YMatrix1,'LineWidth',2,'Parent',axes1);
set(plot1(1),'DisplayName','Frequency');
set(plot1(2),'DisplayName','#Operator','LineStyle','--');
set(plot1(3),'DisplayName','#Input','LineStyle','-.');

% 创建 xlabel
xlabel('Fuzzy rule ID');

% 创建 ylabel
ylabel('Count');

% Create title
title(tmp_tt);

% 取消以下行的注释以保留坐标轴的 X 范围
xlim(axes1,[1 50]);
box(axes1,'on');
% 设置其余坐标轴属性
set(axes1,'FontSize',12,'FontWeight','bold','YTick',[0 2 4 6 8 10],'XTick',...
    [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50]);
% 创建 legend
legend(axes1,'show');

