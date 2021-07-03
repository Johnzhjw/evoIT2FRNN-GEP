function genfig3D(X1, Y1, Z1, X2, Y2, Z2, strAlg, strInst)
%CREATEFIGURE(X1, Y1, X2, Y2)
%  X1:  scatter x
%  Y1:  scatter y
%  X2:  scatter x
%  Y2:  scatter y

%  �� MATLAB �� 27-Jul-2019 09:33:56 �Զ�����

% ���� figure
figure1 = figure;

% ���� axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

% ���� scatter
scatter3(X1,Y1,Z1,'DisplayName','PF',...
    'MarkerFaceColor',[0.831372559070587 0.815686285495758 0.7843137383461],...
    'MarkerEdgeColor',[0.831372559070587 0.815686285495758 0.7843137383461],...
    'Marker','.');

% ���� scatter
scatter3(X2,Y2,Z2,'DisplayName',strAlg,...
    'MarkerFaceColor',[0.749019622802734 0 0.749019622802734],...
    'MarkerEdgeColor',[0.749019622802734 0 0.749019622802734]);

% ���� title
tmp=sprintf('3-Objective %s',strInst);
title(tmp);

% ���� xlabel
xlabel('F1');
% ���� zlabel
zlabel('F3');
% ���� ylabel
ylabel('F2');

% ȡ�������е�ע���Ա���������� X ��Χ
xlim(axes1,[0 ceil(1.5*max(X1))]);
% ȡ�������е�ע���Ա���������� Y ��Χ
ylim(axes1,[0 ceil(1.5*max(Y1))]);
% ȡ�������е�ע���Ա���������� Z ��Χ
zlim(axes1,[0 ceil(1.5*max(Z1))]);
view(axes1,[135 45]);
grid(axes1,'on');

% ������������������
set(axes1,'FontSize',11,'FontWeight','bold');
% ���� legend
legend1 = legend(axes1,'show');
set(legend1,'Location','northeast');

