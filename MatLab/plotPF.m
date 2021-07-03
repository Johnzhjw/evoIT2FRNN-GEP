%%
legendName = cell(length(algs),length(models),2);

for iAlg = 1:length(algs)
    for i = 1:length(models)
        legendName{iAlg,i,1} = sprintf('%s-TRAIN', algs{iAlg});%, models{i});
        legendName{iAlg,i,2} = sprintf('%s-TEST',  algs{iAlg});%, models{i});
    end
end

tmp_cell = cell(1,length(models),2);

tmp_cell{1,1,1} = 'BP-TRAIN';
tmp_cell{1,1,2} = 'BP-TEST';

legendName = [legendName; tmp_cell];

%%
% all_PFs = cell(length(algs),length(modelName),3,length(probs));

for iInst = 1:length(probs)
    tmp = all_ND_PFs(:,:,:,iInst);
    genfig2D(tmp,legendName,probs{iInst});
    fname=sprintf('FIGUREs/PFV_%s.eps',probs{iInst});
    print(fname,'-depsc2','-r300');
    fname=sprintf('FIGUREs/PFV_%s.fig',probs{iInst});
    savefig(fname);
%     close;
end
