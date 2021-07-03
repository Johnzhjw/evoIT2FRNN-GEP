function createfigure_MFutiR(yvector1)
%CREATEFIGURE(YVECTOR1)
%  YVECTOR1:  bar yvector

%  由 MATLAB 于 17-Jan-2021 11:13:56 自动生成

% 创建 figure
figure;

% 创建 axes
axes1 = axes;
hold(axes1,'on');

% 创建 bar
bar(yvector1);

% 创建 xlabel
xlabel('Membership function ID');

% 创建 ylabel
ylabel('Count');

% 取消以下行的注释以保留坐标轴的 X 范围
% xlim(axes1,[0 55]);
box(axes1,'on');
% 设置其余坐标轴属性
set(axes1,'FontWeight','bold','XGrid','on','XTick',...
    [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55],...
    'YGrid','on','YTick',[0 1 2 3 4 5 6]);
