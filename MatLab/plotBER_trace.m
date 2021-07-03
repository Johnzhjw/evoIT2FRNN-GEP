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
            legendName{iiii,i,j} = strrep(legendName{iiii,i,j},'PCMLPSO-A','PCMLPSO');
        end
    end
end

%%

for iInst = 1:length(probs)
    strInst=sprintf('%s',probs{iInst});
%     FEs4grp=0;
    iters=zeros(1,NTRACE+1);
    for I = 1
        for i = 1:(NTRACE+1)
            iters(I,i) = (i - 1) * maxFEs / NTRACE;
        end
    end
%     for I = 1:9
%         for i = 1:26
%             iters(I,i) = FEs4grp + (i-1)*(maxFEs-FEs4grp)/25;
%         end
%     end
%             iters(4,:)=[];
%             tmpMean(4,:)=[];
%             tmpStd(4,:)=[];

% all_mean_over_trace = zeros(length(algs), length(modelName), 3, length(probs), nObj, NTRACE+1);
% all_min_over_trace = zeros(length(algs), length(modelName), 3, length(probs), nObj, NTRACE+1);

    tmp_data = all_mean_over_trace(:,:,:,iInst,1,:);
    tmpStr = 'f_1';
    genfigBER_trace(iters, tmp_data, strInst, legendName, tmpStr);
    fname=sprintf('FIGUREs/F1_%s_%s.eps',strInst,curAlg);
    print(fname,'-depsc2','-r300');
    fname=sprintf('FIGUREs/F1_%s_%s.fig',strInst,curAlg);
    savefig(fname);

    tmp_data = all_mean_over_trace(:,:,:,iInst,2,:);
    tmpStr = 'f_2';
    genfigBER_trace(iters, tmp_data, strInst, legendName, tmpStr);
    fname=sprintf('FIGUREs/F2_%s_%s.eps',strInst,curAlg);
    print(fname,'-depsc2','-r300');
    fname=sprintf('FIGUREs/F2_%s_%s.fig',strInst,curAlg);
    savefig(fname);

    tmp_data = all_mean_over_trace(:,:,:,iInst,3,:);
    tmpStr = 'f_3';
    genfigBER_trace(iters, tmp_data, strInst, legendName, tmpStr);
    fname=sprintf('FIGUREs/F3_%s_%s.eps',strInst,curAlg);
    print(fname,'-depsc2','-r300');
    fname=sprintf('FIGUREs/F3_%s_%s.fig',strInst,curAlg);
    savefig(fname);

%     tmp_data = all_mean_over_trace(:,:,:,:,3,:);
%     tmpStr = 'F1';
%     genfigBER_trace(iters, tmp_data, strInst, legendName, tmpStr);
%     fname=sprintf('FIGUREs/minSPE_%s.eps',allprobnames{iInst});
%     print(fname,'-depsc2','-r300');
%     fname=sprintf('FIGUREs/minSPE_%s.fig',allprobnames{iInst});
%     savefig(fname);
%    close;
end
