%%
legendName = cell(length(algs),length(models),4);

for iiii = 1:lenINDs
    iAlg = tarINDs(iiii);
    for i = length(models)
        %legendName{iiii,i,1} = sprintf('%s-%s-TRAIN', algs{iAlg}, models{i});
        %legendName{iiii,i,2} = sprintf('%s-%s-TEST',  algs{iAlg}, models{i});
        legendName{iiii,i,1} = sprintf('%s-TRAIN',      algs{iAlg});
        legendName{iiii,i,2} = sprintf('%s-TEST',       algs{iAlg});
        legendName{iiii,i,3} = sprintf('%s-BOTH',      algs{iAlg});
        legendName{iiii,i,4} = sprintf('%s-VALIDATION', algs{iAlg});
        for j = 1:4
            legendName{iiii,i,j} = strrep(legendName{iiii,i,j},'0APCMLIA-ADE','PCMLIA-ADE-2L');
            legendName{iiii,i,j} = strrep(legendName{iiii,i,j},'0PCMLIAcur','PCMLIA-ADE-2L-cur');
            legendName{iiii,i,j} = strrep(legendName{iiii,i,j},'0PCMLEA','PCMLEA');
            legendName{iiii,i,j} = strrep(legendName{iiii,i,j},'PCMLIA-ADE-','');
            legendName{iiii,i,j} = strrep(legendName{iiii,i,j},'PCMLIA-ADE-w-','');
            legendName{iiii,i,j} = strrep(legendName{iiii,i,j},'PCMLIA-ADE-Pseu-','');
        end
    end
end

%%

for iInst = 1:length(probs)
    strInst=sprintf('%s',probs{iInst});

    strInst=strrep(sprintf('%s',strInst), '_', ',');
    strInst=strrep(sprintf('%s',strInst), 'evoMobileSink,GEP,only,', '');
    strInst=strrep(sprintf('%s',strInst), 'evoMobileSink,FRNN,flagT2,0,', '');
    strInst=strrep(sprintf('%s',strInst), 'evoMobileSink,FRNN,', '');
    strInst=strrep(sprintf('%s',strInst), 'NumSINK,1', '{\it{N}}_{sink}=1');
    strInst=strrep(sprintf('%s',strInst), 'NumSINK,2', '{\it{N}}_{sink}=2');
    strInst=strrep(sprintf('%s',strInst), 'NumSINK,5', '{\it{N}}_{sink}=5');
    strInst=strrep(sprintf('%s',strInst), 'fAdaptTHrn,0', '{\it{F}}^{adap}_{rsd}=False');
    strInst=strrep(sprintf('%s',strInst), 'fAdaptTHrn,1', '{\it{F}}^{adap}_{rsd}=True');
    strInst=strrep(sprintf('%s',strInst), ',PosType,0,dMove,50', '');
    if contains(sprintf('%s',strInst), 'GRID,10,')
        strInst=sprintf('%s,%s', strInst, '{\it{F}}_{grid}');
        strInst=strrep(sprintf('%s',strInst), 'GRID,10,', '');
    end
    strInst=strrep(sprintf('%s',strInst), 'hungaryChickenpox', 'HungarianChickenpox');
        
    iters=zeros(1,NTRACE+1);
    for I = 1
        for i = 1:(NTRACE+1)
            iters(I,i) = (i - 1) * maxFEs / NTRACE;
        end
    end

    %
    tarINDs = 17:24;
    tmp_data = all_mean_over_trace2(:,:,:,iInst,1,:);
    tmpStr = 'f_1';
    genfigBER_trace(iters, tmp_data, tarINDs, strInst, legendName, tmpStr, 0, 1);
    fname=sprintf('FIGUREs/F1_prob%d.eps',iInst);
    print(fname,'-depsc2','-r300');
    fname=sprintf('FIGUREs/F1_prob%d.fig',iInst);
    savefig(fname);

    tmp_data = all_mean_over_trace2(:,:,:,iInst,2,:);
    tmpStr = 'f_2';
    genfigBER_trace(iters, tmp_data, tarINDs, strInst, legendName, tmpStr, 0, 0);
    fname=sprintf('FIGUREs/F2_prob%d.eps',iInst);
    print(fname,'-depsc2','-r300');
    fname=sprintf('FIGUREs/F2_prob%d.fig',iInst);
    savefig(fname);

    tmp_data = all_mean_over_trace2(:,:,:,iInst,3,:);
    tmpStr = 'f_3';
    genfigBER_trace(iters, tmp_data, tarINDs, strInst, legendName, tmpStr, 0, 0);
    fname=sprintf('FIGUREs/F3_prob%d.eps',iInst);
    print(fname,'-depsc2','-r300');
    fname=sprintf('FIGUREs/F3_prob%d.fig',iInst);
    savefig(fname);
    %
    tarINDs = 25:32;
    tmp_data = all_mean_over_trace2(:,:,:,iInst,1,:);
    tmpStr = 'f_1';
    genfigBER_trace(iters, tmp_data, tarINDs, strInst, legendName, tmpStr, 0, 1);
    fname=sprintf('FIGUREs/F1_T1_prob%d.eps',iInst);
    print(fname,'-depsc2','-r300');
    fname=sprintf('FIGUREs/F1_T1_prob%d.fig',iInst);
    savefig(fname);

    tmp_data = all_mean_over_trace2(:,:,:,iInst,2,:);
    tmpStr = 'f_2';
    genfigBER_trace(iters, tmp_data, tarINDs, strInst, legendName, tmpStr, 0, 0);
    fname=sprintf('FIGUREs/F2_T1_prob%d.eps',iInst);
    print(fname,'-depsc2','-r300');
    fname=sprintf('FIGUREs/F2_T1_prob%d.fig',iInst);
    savefig(fname);

    tmp_data = all_mean_over_trace2(:,:,:,iInst,3,:);
    tmpStr = 'f_3';
    genfigBER_trace(iters, tmp_data, tarINDs, strInst, legendName, tmpStr, 0, 0);
    fname=sprintf('FIGUREs/F3_T1_prob%d.eps',iInst);
    print(fname,'-depsc2','-r300');
    fname=sprintf('FIGUREs/F3_T1_prob%d.fig',iInst);
    savefig(fname);
end
