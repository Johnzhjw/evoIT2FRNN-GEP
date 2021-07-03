function genfig3D(X1, Y1, Z1, X2, Y2, Z2, strAlg, strInst)
%CREATEFIGURE(X1, Y1, X2, Y2)
%  X1:  scatter x
%  Y1:  scatter y
%  X2:  scatter x
%  Y2:  scatter y

%  由 MATLAB 于 27-Jul-2019 09:33:56 自动生成

% 创建 figure
figure1 = figure;

% 创建 axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

% 创建 scatter
scatter3(X1,Y1,Z1,'DisplayName','PF',...
    'MarkerFaceColor',[0.831372559070587 0.815686285495758 0.7843137383461],...
    'MarkerEdgeColor',[0.831372559070587 0.815686285495758 0.7843137383461],...
    'Marker','.');

% 创建 scatter
scatter3(X2,Y2,Z2,'DisplayName',strAlg,...
    'MarkerFaceColor',[0.749019622802734 0 0.749019622802734],...
    'MarkerEdgeColor',[0.749019622802734 0 0.749019622802734]);

% 创建 title
tmp=sprintf('3-Objective %s',strInst);
title(tmp);

% 创建 xlabel
xlabel('F1');
% 创建 zlabel
zlabel('F3');
% 创建 ylabel
ylabel('F2');

% 取消以下行的注释以保留坐标轴的 X 范围
xlim(axes1,[0 ceil(1.5*max(X1))]);
% 取消以下行的注释以保留坐标轴的 Y 范围
ylim(axes1,[0 ceil(1.5*max(Y1))]);
% 取消以下行的注释以保留坐标轴的 Z 范围
zlim(axes1,[0 ceil(1.5*max(Z1))]);
view(axes1,[135 45]);
grid(axes1,'on');

% 设置其余坐标轴属性
set(axes1,'FontSize',11,'FontWeight','bold');
% 创建 legend
legend1 = legend(axes1,'show');
set(legend1,'Location','northeast');

