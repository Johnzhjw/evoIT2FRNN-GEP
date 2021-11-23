%%
table_statistics_FN_mean = zeros(length(probs),length(algs)*3);
table_statistics_FN_std  = zeros(length(probs),length(algs)*3);
table_statistics_RN_mean = zeros(length(probs),length(algs)*3);
table_statistics_RN_std  = zeros(length(probs),length(algs)*3);

for iAlg = 1:length(algs)
    for i=1:length(modelName)
        for j=1:length(probs)
            for k=1:1
                tmp_nonDom=[];
                tmp_PF=[];
                switch k
                    case 1
                        fprintf('PF_%s_%s_%s_TRAIN\n',...
                            algs{iAlg}, modelName{i}, probName{iAlg}{j});
                    case 2
                        fprintf('PF_%s_%s_%s_TEST\n',...
                            algs{iAlg}, modelName{i}, probName{iAlg}{j});
                    case 3
                        fprintf('PF_%s_%s_%s_FINAL\n',...
                            algs{iAlg}, modelName{i}, probName{iAlg}{j});
                    case 4
                        fprintf('PF_%s_%s_%s_VALIDATION\n',...
                            algs{iAlg}, modelName{i}, probName{iAlg}{j});
                end
                fprintf('Run - ');
                fprintf('\n');
                %
                tmp_spl = split(probName{iAlg}{j},'_');
                tmp_spl2= split(algs{iAlg},'-');
                if contains(tmp_spl{end-1}, 'Stock')
                    tmp_tt = sprintf('%s %s.HK', tmp_spl2{end}, strrep(tmp_spl{end},'hungary','Hungarian'));
                else
                    tmp_tt = sprintf('%s %s', tmp_spl2{end}, strrep(tmp_spl{end},'hungary','Hungarian'));
                end
                S_all = all_statistics{iAlg,i,k,j};
                %
                tmp_dt = [S_all{2}; S_all{3}; S_all{4}];
                table_statistics_FN_mean(j,(iAlg-1)*3+1:iAlg*3) = mean(tmp_dt,2);
                table_statistics_FN_std(j,(iAlg-1)*3+1:iAlg*3)  = std(tmp_dt,0,2);
                createfigure_FuzzyRule(tmp_dt', tmp_tt, 'Fuzzy node ID');
                fname = sprintf('FIGUREs/xprs_alg%d_prb%d.eps', iAlg, j);
                print(fname,'-depsc2','-r300');
                fname = sprintf('FIGUREs/xprs_alg%d_prb%d.fig', iAlg, j);
                savefig(fname);
                %
                tmp_dt = [S_all{5}; S_all{6}; S_all{7}];
                table_statistics_RN_mean(j,(iAlg-1)*3+1:iAlg*3) = mean(tmp_dt,2);
                table_statistics_RN_std(j,(iAlg-1)*3+1:iAlg*3)  = std(tmp_dt,0,2);
                createfigure_FuzzyRule(tmp_dt', tmp_tt, 'Rough node ID');
                fname = sprintf('FIGUREs/xprs_rn_alg%d_prb%d.eps', iAlg, j);
                print(fname,'-depsc2','-r300');
                fname = sprintf('FIGUREs/xprs_rn_alg%d_prb%d.fig', iAlg, j);
                savefig(fname);
            end
        end
    end
end

