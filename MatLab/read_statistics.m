%% 
all_statistics = cell(length(algs),length(modelName),length(probs));

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
                for n=1:1
                    fname = sprintf('FUN_%s_', probName{iAlg}{j});
                    ffile = sprintf('%s/%s/tmpFile/', mainfils{iAlg}, fnms{iAlg});
                    fname1 = dir([ffile '*']);
                    tarStr = fname;
                    tmp_cnt = 0;
                    for iName = 1:length(fname1)
                        tmp_fnm = sprintf('%s_', fname1(iName).name);
                        if contains(tmp_fnm, tarStr) && contains(tmp_fnm, '_statistics')
                            tmp_cnt = tmp_cnt + 1;
                            tarName = fname1(iName).name;
                        end
                    end
                    assert(tmp_cnt==1);
                    fprintf('%d ',n);
                    fname2 = sprintf('%s%s', ffile, tarName);
                    tmp   = fun_read_data(fname2);
                end
                all_statistics{iAlg,i,k,j} = tmp;
                fprintf('\n');
            end
        end
    end
end

