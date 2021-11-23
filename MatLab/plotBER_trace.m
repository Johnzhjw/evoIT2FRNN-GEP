%%
legendName = cell(length(algs),length(models),4);

for iiii = 1:lenINDs
    iAlg = tarINDs(iiii);
    for i = length(models)
        %legendName{iiii,i,1} = sprintf('%s-%s-TRAIN', algs{iAlg}, models{i});
        %legendName{iiii,i,2} = sprintf('%s-%s-TEST',  algs{iAlg}, models{i});
        legendName{iiii,i,1} = sprintf('%s-TRAIN',      algs{iAlg});
        legendName{iiii,i,2} = sprintf('%s-TEST',       algs{iAlg});
        legendName{iiii,i,3} = sprintf('%s-FINAL',      algs{iAlg});
        legendName{iiii,i,4} = sprintf('%s-VALIDATION', algs{iAlg});
        for j = 1:4
            legendName{iiii,i,j} = strrep(legendName{iiii,i,j},'PCMLPSO-A','PCMLPSO');
        end
    end
end

%%

titles = {
    'GEP'
    %'IT2FRNN'
    %'FRNN'
};

'evoMobileSink_GEP_only_GRID_10_NumSINK_1_fAdaptTHrn_0_PosType_0_dMove_50';
'evoMobileSink_FRNN_GRID_10_NumSINK_1_fAdaptTHrn_0_PosType_0_dMove_50';
'evoMobileSink_FRNN_flagT2_0_GRID_10_NumSINK_1_fAdaptTHrn_0_PosType_0_dMove_50';

nAlgs_tmp = 1; % 3;

for iInst = 1:1 % 3 %1:length(probs)
    inds = iInst:nAlgs_tmp:length(probs);
    inds = [1 2 3 4 5 6 7 9 10 11];
    strInst=probs(inds);
    for i = 1:length(strInst)
        strInst{i}=strrep(sprintf('%s',strInst{i}), '_', ',');
        strInst{i}=strrep(sprintf('%s',strInst{i}), 'evoMobileSink,GEP,only,', '');
        strInst{i}=strrep(sprintf('%s',strInst{i}), 'evoMobileSink,FRNN,flagT2,0,', '');
        strInst{i}=strrep(sprintf('%s',strInst{i}), 'evoMobileSink,FRNN,', '');
        strInst{i}=strrep(sprintf('%s',strInst{i}), 'NumSINK,1', '{\it{N}}_{sink}=1');
        strInst{i}=strrep(sprintf('%s',strInst{i}), 'NumSINK,2', '{\it{N}}_{sink}=2');
        strInst{i}=strrep(sprintf('%s',strInst{i}), 'NumSINK,5', '{\it{N}}_{sink}=5');
        strInst{i}=strrep(sprintf('%s',strInst{i}), 'fAdaptTHrn,0', '{\it{F}}^{adap}_{rsd}=False');
        strInst{i}=strrep(sprintf('%s',strInst{i}), 'fAdaptTHrn,1', '{\it{F}}^{adap}_{rsd}=True');
        strInst{i}=strrep(sprintf('%s',strInst{i}), ',PosType,0,dMove,50', '');
        if contains(sprintf('%s',strInst{i}), 'GRID,10,')
            strInst{i}=sprintf('%s,%s', strInst{i}, '{\it{F}}_{grid}');
            strInst{i}=strrep(sprintf('%s',strInst{i}), 'GRID,10,', '');
        end
    end
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

    tmp_data = all_mean_over_trace2(:,:,1,inds,1,:);
    tmpStr = 'f_1';
    genfigBER_trace_MobSink(iters, tmp_data, titles{iInst}, strInst, tmpStr, 1);
    fname=sprintf('FIGUREs/F1_TRAIN_%s_%s.eps',titles{iInst});
    print(fname,'-depsc2','-r300');
    fname=sprintf('FIGUREs/F1_TRAIN_%s_%s.fig',titles{iInst});
    savefig(fname);

    tmp_data = all_mean_over_trace2(:,:,2,inds,1,:);
    tmpStr = 'f_1';
    genfigBER_trace_MobSink(iters, tmp_data, titles{iInst}, strInst, tmpStr, 0);
    fname=sprintf('FIGUREs/F1_TEST_%s_%s.eps',titles{iInst});
    print(fname,'-depsc2','-r300');
    fname=sprintf('FIGUREs/F1_TEST_%s_%s.fig',titles{iInst});
    savefig(fname);

    tmp_data = all_mean_over_trace2(:,:,3,inds,1,:);
    tmpStr = 'f_1';
    genfigBER_trace_MobSink(iters, tmp_data, titles{iInst}, strInst, tmpStr, 0);
    fname=sprintf('FIGUREs/F1_BOTH_%s_%s.eps',titles{iInst});
    print(fname,'-depsc2','-r300');
    fname=sprintf('FIGUREs/F1_BOTH_%s_%s.fig',titles{iInst});
    savefig(fname);

    tmp_data = all_mean_over_trace2(:,:,1,inds,2,:);
    tmpStr = 'f_2';
    genfigBER_trace_MobSink(iters, tmp_data, titles{iInst}, strInst, tmpStr, 0);
    fname=sprintf('FIGUREs/F2_TRAIN_%s_%s.eps',titles{iInst});
    print(fname,'-depsc2','-r300');
    fname=sprintf('FIGUREs/F2_TRAIN_%s_%s.fig',titles{iInst});
    savefig(fname);

    tmp_data = all_mean_over_trace2(:,:,2,inds,2,:);
    tmpStr = 'f_2';
    genfigBER_trace_MobSink(iters, tmp_data, titles{iInst}, strInst, tmpStr, 0);
    fname=sprintf('FIGUREs/F2_TEST_%s_%s.eps',titles{iInst});
    print(fname,'-depsc2','-r300');
    fname=sprintf('FIGUREs/F2_TEST_%s_%s.fig',titles{iInst});
    savefig(fname);

    tmp_data = all_mean_over_trace2(:,:,3,inds,2,:);
    tmpStr = 'f_2';
    genfigBER_trace_MobSink(iters, tmp_data, titles{iInst}, strInst, tmpStr, 0);
    fname=sprintf('FIGUREs/F2_BOTH_%s_%s.eps',titles{iInst});
    print(fname,'-depsc2','-r300');
    fname=sprintf('FIGUREs/F2_BOTH_%s_%s.fig',titles{iInst});
    savefig(fname);

    tmp_data = all_mean_over_trace2(:,:,1,inds,3,:);
    tmpStr = 'f_3';
    genfigBER_trace_MobSink(iters, tmp_data, titles{iInst}, strInst, tmpStr, 0);
    fname=sprintf('FIGUREs/F3_TRAIN_%s_%s.eps',titles{iInst});
    print(fname,'-depsc2','-r300');
    fname=sprintf('FIGUREs/F3_TRAIN_%s_%s.fig',titles{iInst});
    savefig(fname);

    tmp_data = all_mean_over_trace2(:,:,2,inds,3,:);
    tmpStr = 'f_3';
    genfigBER_trace_MobSink(iters, tmp_data, titles{iInst}, strInst, tmpStr, 0);
    fname=sprintf('FIGUREs/F3_TEST_%s_%s.eps',titles{iInst});
    print(fname,'-depsc2','-r300');
    fname=sprintf('FIGUREs/F3_TEST_%s_%s.fig',titles{iInst});
    savefig(fname);

    tmp_data = all_mean_over_trace2(:,:,3,inds,3,:);
    tmpStr = 'f_3';
    genfigBER_trace_MobSink(iters, tmp_data, titles{iInst}, strInst, tmpStr, 0);
    fname=sprintf('FIGUREs/F3_BOTH_%s_%s.eps',titles{iInst});
    print(fname,'-depsc2','-r300');
    fname=sprintf('FIGUREs/F3_BOTH_%s_%s.fig',titles{iInst});
    savefig(fname);

    %
%     tmp_data = all_min_over_trace(:,:,1,inds,1,:);
%     tmpStr = 'f_1';
%     genfigBER_trace_MobSink(iters, tmp_data, titles{iInst}, strInst, tmpStr);
%     fname=sprintf('FIGUREs/F1min_TRAIN_%s_%s.eps',titles{iInst});
%     print(fname,'-depsc2','-r300');
%     fname=sprintf('FIGUREs/F1min_TRAIN_%s_%s.fig',titles{iInst});
%     savefig(fname);
% 
%     tmp_data = all_min_over_trace(:,:,2,inds,1,:);
%     tmpStr = 'f_1';
%     genfigBER_trace_MobSink(iters, tmp_data, titles{iInst}, strInst, tmpStr);
%     fname=sprintf('FIGUREs/F1min_TEST_%s_%s.eps',titles{iInst});
%     print(fname,'-depsc2','-r300');
%     fname=sprintf('FIGUREs/F1min_TEST_%s_%s.fig',titles{iInst});
%     savefig(fname);
% 
%     tmp_data = all_min_over_trace(:,:,3,inds,1,:);
%     tmpStr = 'f_1';
%     genfigBER_trace_MobSink(iters, tmp_data, titles{iInst}, strInst, tmpStr);
%     fname=sprintf('FIGUREs/F1min_BOTH_%s_%s.eps',titles{iInst});
%     print(fname,'-depsc2','-r300');
%     fname=sprintf('FIGUREs/F1min_BOTH_%s_%s.fig',titles{iInst});
%     savefig(fname);

%     tmp_data = all_min_over_trace(:,:,1,inds,2,:);
%     tmpStr = 'f_2';
%     genfigBER_trace_MobSink(iters, tmp_data, titles{iInst}, strInst, tmpStr);
%     fname=sprintf('FIGUREs/F2min_TRAIN_%s_%s.eps',titles{iInst});
%     print(fname,'-depsc2','-r300');
%     fname=sprintf('FIGUREs/F2min_TRAIN_%s_%s.fig',titles{iInst});
%     savefig(fname);
% 
%     tmp_data = all_min_over_trace(:,:,2,inds,2,:);
%     tmpStr = 'f_2';
%     genfigBER_trace_MobSink(iters, tmp_data, titles{iInst}, strInst, tmpStr);
%     fname=sprintf('FIGUREs/F2min_TEST_%s_%s.eps',titles{iInst});
%     print(fname,'-depsc2','-r300');
%     fname=sprintf('FIGUREs/F2min_TEST_%s_%s.fig',titles{iInst});
%     savefig(fname);
% 
%     tmp_data = all_min_over_trace(:,:,1,inds,3,:);
%     tmpStr = 'f_3';
%     genfigBER_trace_MobSink(iters, tmp_data, titles{iInst}, strInst, tmpStr);
%     fname=sprintf('FIGUREs/F3min_TRAIN_%s_%s.eps',titles{iInst});
%     print(fname,'-depsc2','-r300');
%     fname=sprintf('FIGUREs/F3min_TRAIN_%s_%s.fig',titles{iInst});
%     savefig(fname);
% 
%     tmp_data = all_min_over_trace(:,:,2,inds,3,:);
%     tmpStr = 'f_3';
%     genfigBER_trace_MobSink(iters, tmp_data, titles{iInst}, strInst, tmpStr);
%     fname=sprintf('FIGUREs/F3min_TEST_%s_%s.eps',titles{iInst});
%     print(fname,'-depsc2','-r300');
%     fname=sprintf('FIGUREs/F3min_TEST_%s_%s.fig',titles{iInst});
%     savefig(fname);

end

% %%
% 
% for iInst = 1:length(probs)
%     strInst=strrep(sprintf('%s',probs{iInst}), '_', '-');
% %     FEs4grp=0;
%     iters=zeros(1,NTRACE+1);
%     for I = 1
%         for i = 1:(NTRACE+1)
%             iters(I,i) = (i - 1) * maxFEs / NTRACE;
%         end
%     end
% %     for I = 1:9
% %         for i = 1:26
% %             iters(I,i) = FEs4grp + (i-1)*(maxFEs-FEs4grp)/25;
% %         end
% %     end
% %             iters(4,:)=[];
% %             tmpMean(4,:)=[];
% %             tmpStd(4,:)=[];
% 
% % all_mean_over_trace = zeros(length(algs), length(modelName), 3, length(probs), nObj, NTRACE+1);
% % all_min_over_trace = zeros(length(algs), length(modelName), 3, length(probs), nObj, NTRACE+1);
% 
%     tmp_data = all_mean_over_trace(:,:,:,iInst,1,:);
%     tmpStr = 'f_1';
%     genfigBER_trace(iters, tmp_data, strInst, legendName, tmpStr);
%     fname=sprintf('FIGUREs/F1_%s_%s.eps',strInst,curAlg);
%     print(fname,'-depsc2','-r300');
%     fname=sprintf('FIGUREs/F1_%s_%s.fig',strInst,curAlg);
%     savefig(fname);
% 
%     tmp_data = all_mean_over_trace(:,:,:,iInst,2,:);
%     tmpStr = 'f_2';
%     genfigBER_trace(iters, tmp_data, strInst, legendName, tmpStr);
%     fname=sprintf('FIGUREs/F2_%s_%s.eps',strInst,curAlg);
%     print(fname,'-depsc2','-r300');
%     fname=sprintf('FIGUREs/F2_%s_%s.fig',strInst,curAlg);
%     savefig(fname);
% 
%     tmp_data = all_mean_over_trace(:,:,:,iInst,3,:);
%     tmpStr = 'f_3';
%     genfigBER_trace(iters, tmp_data, strInst, legendName, tmpStr);
%     fname=sprintf('FIGUREs/F3_%s_%s.eps',strInst,curAlg);
%     print(fname,'-depsc2','-r300');
%     fname=sprintf('FIGUREs/F3_%s_%s.fig',strInst,curAlg);
%     savefig(fname);
% 
% %     tmp_data = all_mean_over_trace(:,:,:,:,3,:);
% %     tmpStr = 'F1';
% %     genfigBER_trace(iters, tmp_data, strInst, legendName, tmpStr);
% %     fname=sprintf('FIGUREs/minSPE_%s.eps',allprobnames{iInst});
% %     print(fname,'-depsc2','-r300');
% %     fname=sprintf('FIGUREs/minSPE_%s.fig',allprobnames{iInst});
% %     savefig(fname);
% %    close;
% end
