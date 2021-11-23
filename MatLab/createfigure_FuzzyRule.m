function createfigure_FuzzyRule(YMatrix1, tmp_tt, xstr)
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
xlabel(xstr);

% 创建 ylabel
ylabel('Count');

% Create title
title(tmp_tt);

% 取消以下行的注释以保留坐标轴的 X 范围
xlim(axes1,[1 size(YMatrix1,1)]);
box(axes1,'on');
% 设置其余坐标轴属性
Ymax = max(max(YMatrix1));
Ymax = Ymax+mod(5-mod(Ymax,5),5);
Ygap = Ymax/5;

set(axes1,'FontSize',12,'FontWeight','bold','YTick',0:Ygap:Ymax,'XTick',1:size(YMatrix1,1));

set(gcf,'unit','centimeters','position',[10 5 max(size(YMatrix1,1)/50*40,15) 5])

% 创建 legend
legend(axes1,'show','Location','best');

