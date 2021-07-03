%% 
all_PFs_4_run_trace    = cell(length(algs),length(modelName),4,length(probs),nRun,NTRACE+1);

for iAlg = 1:length(algs)
    for i=1:length(modelName)
        for j=1:length(probs)
            for k=1:3
                tmp_nonDom=[];
                tmp_PF=[];
                for n=1:nRun
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
                    fname = sprintf('%s_FUN_%s_*',...
                                        filestrs{iAlg}, probName{iAlg}{j});
                    ffile = sprintf('%s/%s/%s/trace/', mainfils{iAlg}, fnms{iAlg}, PFfilestrs{iAlg});
                    fname = dir([ffile fname]);
                    fprintf('%d \n',n);
                    tarStr = sprintf('_RUN%d_',n);
                    names_train = {};
                    names_test  = {};
                    names_final = {};
                    names_validation = {};
                    i_train = 0;
                    i_test  = 0;
                    i_final = 0;
                    i_validation = 0;
                    for iName = 1:length(fname)
                        if strfind(fname(iName).name, tarStr)
                            if strfind(fname(iName).name, 'TRAIN')
                                i_train = i_train + 1;
                                names_train{i_train,1} = sprintf('%s_', fname(iName).name);
                            elseif strfind(fname(iName).name, 'TEST')
                                i_test = i_test + 1;
                                names_test{i_test,1} = sprintf('%s_', fname(iName).name);
                            elseif strfind(fname(iName).name, 'VALIDATION')
                                i_validation = i_validation + 1;
                                names_validation{i_validation,1} = sprintf('%s_', fname(iName).name);
                            else
                                i_final = i_final + 1;
                                names_final{i_final,1} = sprintf('%s_', fname(iName).name);
                            end
                        end
                    end
                    switch k
                        case 1
                            cur_names = names_train;
                            cur_len   = i_train;
                        case 2
                            cur_names = names_test;
                            cur_len   = i_test;
                        case 3
                            cur_names = names_final;
                            cur_len   = i_final;
                        case 3
                            cur_names = names_validation;
                            cur_len   = i_validation;
                    end
                    for iTrace = 1 : NTRACE
                        tarStr = sprintf('key%d_', iTrace-1);
                        fprintf('%s ', tarStr);
                        for ijk = 1:cur_len
                            if strfind(cur_names{ijk}, tarStr)
                                theName = cur_names{ijk};
                                theName(end)='';
                            end
                        end
%                         if iTrace > 22
%                             tmtmtmtppp = 0;
%                         end
                        fprintf('%s ', theName);
                        fname2 = sprintf('%s%s', ffile, theName);
                        tmp   = importdata(fname2);
                        nPop  = size(tmp,1);
                        nObj  = size(tmp,2);
                        
                        all_PFs_4_run_trace{iAlg,i,k,j,n,iTrace} = tmp;
                    end
                    tmp_ind_st = (n-1)*nPop+1;
                    tmp_ind_fn = n*nPop;
                    all_PFs_4_run_trace{iAlg,i,k,j,n,NTRACE+1} = all_PFs{iAlg,i,k,j}(tmp_ind_st:tmp_ind_fn,1:nObj);
                    fprintf('\n');
               end
            end
        end
    end
end


