%%
legendName = cell(length(algs),length(models));

for iAlg = 1:length(algs)
    for i = 1:length(models)
        legendName{iAlg,i} = sprintf('%s-%s', algs{iAlg}, models{i});
    end
end

tmp_cell = cell(1,length(models));

tmp_cell{1,1} = sprintf('%s-%s', 'BP', models{1});

legendName = [legendName; tmp_cell];

%%
% all_PFs = cell(length(algs),length(modelName),3,length(probs));

for iInst = 1:length(probs)
    tmp = all_nd_prc(:,:,iInst);
    genfig2D_prc(tmp,legendName,probs{iInst});
%    fname=sprintf('FIGUREs/PFV_prc_STOCK%02d_PCMLIA_ADE.eps',iInst);
    fname=sprintf('FIGUREs/PFV_prc_STOCK%02d_Model5_2.eps',iInst);
%    fname=sprintf('FIGUREs/PFV_prc_STOCK%02d_NO_XOR.eps',iInst);
%    fname=sprintf('FIGUREs/PFV_prc_STOCK%02d_SHADE.eps',iInst);
    print(fname,'-depsc2','-r300');
%    fname=sprintf('FIGUREs/PFV_prc_STOCK%02d_PCMLIA_ADE.fig',iInst);
    fname=sprintf('FIGUREs/PFV_prc_STOCK%02d_Model5_2.fig',iInst);
%    fname=sprintf('FIGUREs/PFV_prc_STOCK%02d_NO_XOR.fig',iInst);
%    fname=sprintf('FIGUREs/PFV_prc_STOCK%02d_SHADE.fig',iInst);
    savefig(fname);
%     close;
end


