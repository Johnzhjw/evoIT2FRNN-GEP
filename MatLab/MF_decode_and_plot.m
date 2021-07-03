function [ output_args ] = MF_decode_and_plot( MFparas, numInputs, numMf, stockCode )
%%
%   MFparas   - parameters for membership functions
%   numInputs - number of inputs
%   numMf     - number of membership functions for each input
%

%%
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

line_types = {
    '-'
    '--'
    ':'
    '-.'
};

%%
for iIn = 1 : numInputs
    p_fg = figure;
    p_ax = axes('Parent',p_fg);
    hold(p_ax,'on');
    x = -2:0.02:2;
    for iMf = 1 : numMf
        iRow = (iIn - 1) * numMf + iMf;
        type_MF = MFparas(iRow,3);
        sigma1 = MFparas(iRow,4);
        c1     = MFparas(iRow,5);
        sigma2 = MFparas(iRow,6);
        c2     = MFparas(iRow,7);
        gamma  = MFparas(iRow,8);
        c3     = MFparas(iRow,9);
        a1     = MFparas(iRow,10);
        a2     = MFparas(iRow,11);
        if type_MF == 0
            y = gaussmf(x,[sigma1 c1]);
            nm_MF_l = 'Gaussian lower';
            nm_MF_u = 'Gaussian upper';
        elseif type_MF == 1
            y = sigmf(x,[gamma c3]);
            nm_MF_l = 'Sigmoid lower';
            nm_MF_u = 'Sigmoid upper';
        elseif type_MF == 2
            y = gauss2mf(x,[sigma1 c1 sigma2 c2]);
            nm_MF_l = 'Gaussian combination lower';
            nm_MF_u = 'Gaussian combination upper';
        else
            disp('Unknown MF type.');
        end
        yh = (1 - (1-y).^a1).^(1/a1);
        yl = (1 - (1-y).^a2).^(1/a2);
        %
        plot(...
            x,yl,'LineWidth',2,...
            'LineStyle',line_types{iMf},...
            'DisplayName',nm_MF_l,...
            'color',colors{iMf}...
            );
        plot(...
            x,yh,'LineWidth',2.5,...
            'LineStyle',line_types{iMf},...
            'DisplayName',nm_MF_u,...
            'color',colors{iMf}...
            );
    end
    %
    xlim(p_ax,[-2 2]);
    xlabel('Input values');
    tmp = sprintf('Input feature %d', iIn);
    title(tmp);
    ylabel('Membership degrees');
    box(p_ax,'on');
    p_legend = legend(p_ax,'show');
    set(p_legend,'Location','best');
    set(p_ax,'FontSize',11,'FontWeight','bold','XGrid','on',...
        'YGrid','on','YMinorTick','on');
    %
    fname = sprintf('FIGUREs/Stock_%s_MF_input_%d.eps',stockCode,iIn);
    print(fname,'-depsc2','-r300');
    fname = sprintf('FIGUREs/Stock_%s_MF_input_%d.fig',stockCode,iIn);
    savefig(fname);
end

end

