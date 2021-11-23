function createfigure_FuzzyRule(YMatrix1, tmp_tt, xstr)
%CREATEFIGURE(YMATRIX1)
%  YMATRIX1:  y ���ݵľ���

%  �� MATLAB �� 17-Jan-2021 17:16:28 �Զ�����

% ���� figure
figure1 = figure;

% ���� axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

% ʹ�� plot �ľ������봴������
plot1 = plot(YMatrix1,'LineWidth',2,'Parent',axes1);
set(plot1(1),'DisplayName','Frequency');
set(plot1(2),'DisplayName','#Operator','LineStyle','--');
set(plot1(3),'DisplayName','#Input','LineStyle','-.');

% ���� xlabel
xlabel(xstr);

% ���� ylabel
ylabel('Count');

% Create title
title(tmp_tt);

% ȡ�������е�ע���Ա���������� X ��Χ
xlim(axes1,[1 size(YMatrix1,1)]);
box(axes1,'on');
% ������������������
Ymax = max(max(YMatrix1));
Ymax = Ymax+mod(5-mod(Ymax,5),5);
Ygap = Ymax/5;

set(axes1,'FontSize',12,'FontWeight','bold','YTick',0:Ygap:Ymax,'XTick',1:size(YMatrix1,1));

set(gcf,'unit','centimeters','position',[10 5 max(size(YMatrix1,1)/50*40,15) 5])

% ���� legend
legend(axes1,'show','Location','best');

